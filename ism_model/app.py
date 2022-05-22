from re import A
import fire
import os
import logging
import logging.config
import multiprocessing as mp
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Union, List

from ism_model.utils.yaml import load_yaml_safe
from ism_model.utils.dataset import Data
import ism_model.config as cfg
from ism_model.model.ism import Model
from ism_model.metrics.metrics import Theil, RMSPE, MAPE, R2, CORR
from ism_model.visual.plot_2d import plot_time_series, plot_metrics, plot_tube_predictions


class Pipeline:
    def __init__(self, run_config: str):
        logging.info("Started the pipeline")
        logging.config.dictConfig(cfg.LOG_DICT_CONFIG)
        self.run_config = run_config
        self.params = load_yaml_safe(yaml_path=self.run_config)

        self.tech_params = self.params["tech"]
        self.data_params = self.params["data"]
        self.model_params = self.params["model"]

        # self.model = Model()

    def load_dataset(self):
        data = Data(
            source_file=os.path.join(
                self.tech_params["data_dir"],
                self.data_params["file"]
            )
        ).load()
        data["p_y"] = data[cfg.GDP_COL]/data[cfg.GDP_COL+" const"]
        data["p_i"] = data[cfg.IMPORT_COL]/data[cfg.IMPORT_COL+" const"]
        data["p_j"] = data[cfg.INVESTMENTS_COL]/data[cfg.INVESTMENTS_COL+" const"]
        data["p_e"] = data[cfg.EXPORT_COL]/data[cfg.EXPORT_COL+" const"]
        data["pi_i"] = data["p_i"]/data["p_y"]
        data["pi_j"] = data["p_j"]/data["p_y"]
        data["pi_e"] = data["p_e"]/data["p_y"]
        return data

    @staticmethod
    def save_tube_predictions(data: np.array, name: str) -> None:
        df = pd.DataFrame(
            data=data,
            columns=[str(i) for i in range(data.shape[1])]
        )
        df.to_csv(os.path.join("predictions", f"{name}_tube_predictions.csv"), index=False)

    def inference(self, params_df: pd.DataFrame,  # data_df: pd.DataFrame,
                  return_predictions: bool = False) -> List[List[Union[float, int]]]:
        res = []
        if return_predictions:
            Y_preds_all = []
            I_preds_all = []
            E_preds_all = []
        for ind in params_df.index:
            Y_preds = []
            I_preds = []
            E_preds = []
            K_t = self.dataset[cfg.GDP_COL+" const"][0]/params_df["alpha_k"][ind]
            K_t_next = self.dataset[cfg.GDP_COL+" const"][0]/params_df["alpha_k"][ind]
            Y_t = self.dataset[cfg.GDP_COL+" const"][0]
            for t in range(self.dataset.shape[0]):
                K_t = K_t_next
                Y_t = self.model.calc_gdp(
                    Y_0=self.dataset[cfg.GDP_COL+" const"][0],
                    a=params_df["a"][ind],
                    b=params_df["b"][ind],
                    L=self.dataset[cfg.LABOUR_COL+" preds"][t],
                    L_0=self.dataset[cfg.LABOUR_COL+" preds"][0],
                    K=K_t,
                    K_0=self.dataset[cfg.GDP_COL+" const"][0]/params_df["alpha_k"][ind]
                )
                I_t = self.model.calc_import(
                    rho=self.dataset["rho"][t],
                    delta=self.dataset["delta"][t],
                    Y=Y_t,
                    pi_i=self.dataset["pi_i preds"][t]
                )
                E_t = self.model.calc_export(
                    delta=self.dataset["delta"][t],
                    Y=Y_t,
                    pi_e=self.dataset["pi_e preds"][t]
                )
                J_t = self.model.calc_investments(
                    sigma=self.dataset["sigma"][t],
                    rho=self.dataset["rho"][t],
                    delta=self.dataset["delta"][t],
                    Y=Y_t,
                    pi_j=self.dataset["pi_j preds"][t]
                )
                Y_preds.append(Y_t)
                I_preds.append(I_t)
                E_preds.append(E_t)
                K_t_next = self.model.calc_capital(
                    J=J_t,
                    mu=params_df["mu"][ind],
                    K=K_t
                )
            corr_metrics = [
                CORR(Y_preds, self.dataset[cfg.GDP_COL+" const"].to_list()),
                CORR(I_preds, self.dataset[cfg.IMPORT_COL+" const"].to_list()),
                CORR(E_preds, self.dataset[cfg.EXPORT_COL+" const"].to_list())
            ]
            mape_metrics = [
                MAPE(Y_preds, self.dataset[cfg.GDP_COL+" const"].to_list()),
                MAPE(I_preds, self.dataset[cfg.IMPORT_COL+" const"].to_list()),
                MAPE(E_preds, self.dataset[cfg.EXPORT_COL+" const"].to_list())
            ]
            res.append([np.mean(corr_metrics), np.mean(mape_metrics), ind])
            if return_predictions:
                Y_preds_all.append(Y_preds)
                I_preds_all.append(I_preds)
                E_preds_all.append(E_preds)
        if return_predictions:
            self.save_tube_predictions(np.array(Y_preds_all), "GDP")
            self.save_tube_predictions(np.array(I_preds_all), "Import")
            self.save_tube_predictions(np.array(E_preds_all), "Export")
        return res

    def run(self):
        logging.info("Loading train data")
        self.dataset = self.load_dataset()
        self.dataset = self.dataset[
            (self.dataset[cfg.YEAR_COL] > self.data_params["start_year"]) &
            (self.dataset[cfg.YEAR_COL] <= self.data_params["end_year"])
        ]
        self.dataset = self.dataset[[
            cfg.GDP_COL,
            cfg.GDP_COL+" const",
            cfg.LABOUR_COL,
            cfg.EXPORT_COL,
            cfg.EXPORT_COL+" const",
            cfg.IMPORT_COL,
            cfg.IMPORT_COL+" const",
            cfg.INVESTMENTS_COL,
            cfg.INVESTMENTS_COL+" const",
            "pi_i",
            "pi_j",
            "pi_e"
        ]].reset_index(drop=True)
        self.dataset[cfg.LABOUR_COL] *= 0.6

        logging.info("Initializing the model")
        self.model = Model(L_0=self.dataset[cfg.LABOUR_COL][0], Y_0=self.dataset[cfg.GDP_COL][0])

        logging.info("Fitting labour curve")
        L_0, n = self.model.fit_exponential(self.dataset[cfg.LABOUR_COL].to_list())
        L_preds = [L_0*math.exp(n*t) for t in range(self.dataset.shape[0])]
        self.dataset[cfg.LABOUR_COL+" preds"] = L_preds
        plot_time_series(
            data_y={"Predicted": L_preds, "Real": self.dataset[cfg.LABOUR_COL].to_list()},
            colors=["blue", "green"],
            title="Labour curve",
            x_label="Time",
            y_label="Labour",
            save_path="img/labour_curve.jpeg"
        )

        logging.info("Fitting import price index")
        pi_i_preds = self.model.fit_polinominal(self.dataset["pi_i"].to_list(), 7)
        self.dataset["pi_i preds"] = pi_i_preds
        # a_i, b_i = self.model.fit_exponential(self.dataset["pi_i"].to_list())
        # pi_i_preds = [a_i*math.exp(b_i*t) for t in range(self.dataset.shape[0])]
        plot_time_series(
            data_y={"Predicted": pi_i_preds, "Real": self.dataset["pi_i"].to_list()},
            colors=["blue", "green"],
            title="Import price index curve",
            x_label="Time",
            y_label="Import price index",
            save_path="img/import_price_index_curve.jpeg"
        )

        logging.info("Fitting export price index")
        pi_e_preds = self.model.fit_polinominal(self.dataset["pi_e"].to_list(), 7)
        self.dataset["pi_e preds"] = pi_e_preds
        # a_e, b_e = self.model.fit_exponential(self.dataset["pi_e"].to_list())
        # pi_e_preds = [a_e*math.exp(b_e*t) for t in range(self.dataset.shape[0])]
        plot_time_series(
            data_y={"Predicted": pi_e_preds, "Real": self.dataset["pi_e"].to_list()},
            colors=["blue", "green"],
            title="Export price index curve",
            x_label="Time",
            y_label="Export price index",
            save_path="img/export_price_index_curve.jpeg"
        )

        logging.info("Fitting investments price index")
        pi_j_preds = self.model.fit_polinominal(self.dataset["pi_j"].to_list(), 7)
        self.dataset["pi_j preds"] = pi_j_preds
        # a_j, b_j = self.model.fit_exponential(self.dataset["pi_j"].to_list())
        # pi_j_preds = [a_j*math.exp(b_j*t) for t in range(self.dataset.shape[0])]
        plot_time_series(
            data_y={"Predicted": pi_j_preds, "Real": self.dataset["pi_j"].to_list()},
            colors=["blue", "green"],
            title="Invesments price index curve",
            x_label="Time",
            y_label="Invesments price index",
            save_path="img/invesments_price_index_curve.jpeg"
        )

        logging.info("Calculating sigma, delta and rho")
        sigma = self.model.calc_sigma(
            pi_j_list=self.dataset["pi_j preds"],
            J_list=self.dataset[cfg.INVESTMENTS_COL],
            Y_list=self.dataset[cfg.GDP_COL],
            pi_i_list=self.dataset["pi_i preds"],
            Imp_list=self.dataset[cfg.IMPORT_COL]
        )
        sigma_preds = self.model.fit_coeffs(sigma)
        self.dataset["sigma"] = sigma_preds
        # print(sigma, sigma_preds)
        delta = self.model.calc_delta(
            pi_e_list=self.dataset["pi_e preds"],
            E_list=self.dataset[cfg.EXPORT_COL],
            Y_list=self.dataset[cfg.GDP_COL]
        )
        delta_preds = self.model.fit_coeffs(delta)
        self.dataset["delta"] = delta_preds
        # print(delta, delta_preds)
        rho = self.model.calc_rho(
            pi_i_list=self.dataset["pi_i preds"],
            Imp_list=self.dataset[cfg.IMPORT_COL],
            Y_list=self.dataset[cfg.GDP_COL],
            pi_e_list=self.dataset["pi_e preds"],
            E_list=self.dataset[cfg.EXPORT_COL]
        )
        rho_preds = self.model.fit_coeffs(rho)
        self.dataset["rho"] = rho_preds
        # print(rho, rho_preds)
        # self.model.set_static_params(sigma=sigma_mean, delta=delta_mean, rho=rho_mean)

        logging.info("Creating mesh grid for model's params")
        self.params_set = self.model.create_params_set(
            coeffs=self.model_params["coeffs"],
            n_splits=self.model_params["n_splits"]
        )

        logging.info(f"You will be using {self.model_params['n_threads']} cores out of {mp.cpu_count()} available")
        logging.info(f"A total of {self.params_set.shape[0]} parameters combinations will be tested")
        cores = self.model_params['n_threads']
        params_splitted = np.array_split(self.params_set, cores)
        pool = mp.Pool(self.model_params['n_threads'])
        results = [pool.apply_async(self.inference, args=(df,)).get() for df in params_splitted]
        pool.close()
        results = np.concatenate(results).reshape(-1, 3)
        results = results[results[:, 2].argsort()]
        self.params_set["corr_metrics"] = results[:, 0].reshape(-1, )
        self.params_set["MAPE_metrics"] = results[:, 1].reshape(-1, )
        self.params_set = self.params_set[
            (self.params_set["corr_metrics"] >= 0.5) &
            (self.params_set["MAPE_metrics"] <= 0.5)
        ]
        self.params_set["corr_metrics"] = 1 - self.params_set["corr_metrics"]
        metrics_arr = self.params_set[["corr_metrics", "MAPE_metrics"]].to_numpy().reshape(-1, 2)

        logging.info("Calculating pareto front")
        pareto_mask = self.model.calc_pareto_front(metrics_arr)
        self.params_set["pareto_mask"] = pareto_mask
        pareto_points = self.params_set[self.params_set["pareto_mask"] == True]
        not_pareto_points = self.params_set[self.params_set["pareto_mask"] == False]
        # metrics_arr = [(point[0], point[1]) for point in metrics_arr]
        # shell_arr = np.array(self.model.create_shell(metrics_arr, self.model_params["convex_alpha"]))
        plot_metrics(
            pareto_points_x=pareto_points["corr_metrics"].to_list(),
            pareto_points_y=pareto_points["MAPE_metrics"].to_list(),
            not_pareto_points_x=not_pareto_points["corr_metrics"].to_list(),
            not_pareto_points_y=not_pareto_points["MAPE_metrics"].to_list(),
            title="Metrics",
            x_label="1 - corr_metrics",
            y_label="MAPE_metrics",
            save_path="img/metrics.jpeg"
        )

        logging.info("Calculating preidctions tube using pareto front")
        params_splitted = np.array_split(pareto_points, 1)
        pool = mp.Pool(self.model_params['n_threads'])
        results = [pool.apply_async(self.inference, args=(df, True, )).get() for df in params_splitted]
        pool.close()
        Y_preds = pd.read_csv(os.path.join("predictions", "GDP_tube_predictions.csv"))
        I_preds = pd.read_csv(os.path.join("predictions", "Import_tube_predictions.csv"))
        E_preds = pd.read_csv(os.path.join("predictions", "Export_tube_predictions.csv"))
        plot_tube_predictions(
            trace_real=self.dataset[cfg.GDP_COL+" const"].to_list(),
            traces_predicted=np.array(Y_preds),
            title="GDP tube predictions",
            x_label="Time",
            y_label="GDP",
            save_path="img/GDP_tube_predictions"
        )
        plot_tube_predictions(
            trace_real=self.dataset[cfg.IMPORT_COL+" const"].to_list(),
            traces_predicted=np.array(I_preds),
            title="Import tube predictions",
            x_label="Time",
            y_label="Import",
            save_path="img/Import_tube_predictions"
        )
        plot_tube_predictions(
            trace_real=self.dataset[cfg.EXPORT_COL+" const"].to_list(),
            traces_predicted=np.array(E_preds),
            title="Export tube predictions",
            x_label="Time",
            y_label="Export",
            save_path="img/Export_tube_predictions"
        )

        logging.info("Pipeline done!")


if __name__ == "__main__":
    fire.Fire(Pipeline)
