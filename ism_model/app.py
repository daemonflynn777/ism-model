from re import A
import fire
import os
import logging
import logging.config
import multiprocessing as mp
import pandas as pd
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
from ism_model.visual.plot_2d import plot_time_series, plot_metrics


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

    def inference(self, df: pd.DataFrame) -> List[List[Union[float, int]]]:
        res = []
        for ind in df.index:
            Y_preds = []
            I_preds = []
            E_preds = []
            K_t = self.dataset[cfg.GDP_COL][0]/df["alpha_k"][ind]
            K_t_next = self.dataset[cfg.GDP_COL][0]/df["alpha_k"][ind]
            Y_t = self.dataset[cfg.GDP_COL][0]
            for t in range(self.dataset.shape[0]):
                K_t = K_t_next
                Y_t = self.model.calc_gdp(
                    Y_0=self.dataset[cfg.GDP_COL][0],
                    a=df["a"][ind],
                    b=df["b"][ind],
                    L=self.dataset[cfg.LABOUR_COL+" preds"][t],
                    L_0=self.dataset[cfg.LABOUR_COL+" preds"][0],
                    K=K_t,
                    K_0=self.dataset[cfg.GDP_COL][0]/df["alpha_k"][ind]
                )
                I_t = self.model.calc_import(
                    rho=self.model.rho,
                    delta=self.model.delta,
                    Y=Y_t,
                    pi_i=self.dataset["pi_i preds"][t]
                )
                E_t = self.model.calc_export(
                    delta=self.model.delta,
                    Y=Y_t,
                    pi_e=self.dataset["pi_e preds"][t]
                )
                J_t = self.model.calc_investments(
                    rho=self.model.rho,
                    delta=self.model.delta,
                    Y=Y_t,
                    pi_j=self.dataset["pi_j preds"][t]
                )
                Y_preds.append(Y_t)
                I_preds.append(I_t)
                E_preds.append(E_t)
                K_t_next = self.model.calc_capital(
                    J=J_t,
                    mu=df["mu"][ind],
                    K=K_t
                )
            corr_metrics = [
                CORR(Y_preds, self.dataset[cfg.GDP_COL].to_list()),
                CORR(I_preds, self.dataset[cfg.IMPORT_COL].to_list()),
                CORR(E_preds, self.dataset[cfg.EXPORT_COL].to_list())
            ]
            mape_metrics = [
                MAPE(Y_preds, self.dataset[cfg.GDP_COL].to_list()),
                MAPE(I_preds, self.dataset[cfg.IMPORT_COL].to_list()),
                MAPE(E_preds, self.dataset[cfg.EXPORT_COL].to_list())
            ]
            res.append([np.mean(corr_metrics), np.mean(mape_metrics), ind])
        return res

    def run(self):
        # for el in self.model_params["coeffs"]:
        #     print(el)
        # self.model.create_params_set(self.model_params["coeffs"], self.model_params["n_splits"])
        logging.info("Loading train data")
        self.dataset = self.load_dataset()
        self.dataset = self.dataset[
            (self.dataset[cfg.YEAR_COL] > self.data_params["start_year"]) &
            (self.dataset[cfg.YEAR_COL] <= self.data_params["end_year"])
        ]
        self.dataset = self.dataset[[
            cfg.GDP_COL,
            cfg.LABOUR_COL,
            cfg.EXPORT_COL,
            cfg.IMPORT_COL,
            cfg.INVESTMENTS_COL,
            "pi_i",
            "pi_j",
            "pi_e"
        ]].reset_index(drop=True)
        self.dataset[cfg.LABOUR_COL] *= 0.6
        print(self.dataset.head(10))

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
        pi_i_preds = self.model.fit_polinominal(self.dataset["pi_i"].to_list(), 5)
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
        pi_e_preds = self.model.fit_polinominal(self.dataset["pi_e"].to_list(), 5)
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
        pi_j_preds = self.model.fit_polinominal(self.dataset["pi_j"].to_list(), 5)
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
        sigma_mean, sigma_std = self.model.calc_sigma(
            pi_j_list=self.dataset["pi_j preds"],
            J_list=self.dataset[cfg.INVESTMENTS_COL],
            Y_list=self.dataset[cfg.GDP_COL],
            pi_i_list=self.dataset["pi_i preds"],
            Imp_list=self.dataset[cfg.IMPORT_COL]
        )
        print(sigma_mean)
        delta_mean, delta_std = self.model.calc_delta(
            pi_e_list=self.dataset["pi_e preds"],
            E_list=self.dataset[cfg.EXPORT_COL],
            Y_list=self.dataset[cfg.GDP_COL]
        )
        print(delta_mean)
        rho_mean, rho_std = self.model.calc_rho(
            pi_i_list=self.dataset["pi_i preds"],
            Imp_list=self.dataset[cfg.IMPORT_COL],
            Y_list=self.dataset[cfg.GDP_COL],
            pi_e_list=self.dataset["pi_e preds"],
            E_list=self.dataset[cfg.EXPORT_COL]
        )
        print(rho_mean)
        self.model.set_static_params(sigma=sigma_mean, delta=delta_mean, rho=rho_mean)

        logging.info("Creating mesh grid for model's params")
        self.params_set = self.model.create_params_set(
            coeffs=self.model_params["coeffs"],
            n_splits=self.model_params["n_splits"]
        )

        logging.info(f"You will be using {self.model_params['n_threads']} cores out of {mp.cpu_count()} available")
        rows_per_core = self.params_set.shape[0] // self.model_params['n_threads']
        print(self.params_set.shape[0])
        cores = self.model_params['n_threads']
        # params_splitted = (
        #     self.params_set.iloc[rows_per_core*split:min(rows_per_core*(split+1), self.params_set.shape[0]), :]
        #     for split in range(cores)
        # )
        params_splitted = np.array_split(self.params_set, cores)
        pool = mp.Pool(self.model_params['n_threads'])
        results = [pool.apply_async(self.inference, args=(df,)).get() for df in params_splitted]
        pool.close()
        results = np.concatenate(results).reshape(-1, 3)
        results = results[results[:, 2].argsort()]
        print(results[:5])
        self.params_set["corr_metrics"] = results[:, 0].reshape(-1, )
        self.params_set["MAPE_metrics"] = results[:, 1].reshape(-1, )
        self.params_set = self.params_set[
            (self.params_set["corr_metrics"] >= 0.5) &
            (self.params_set["MAPE_metrics"] <= 0.5)
        ]
        print(self.params_set.sort_values(by=["MAPE_metrics", "corr_metrics"], ascending=[True, False]).head(15))
        plot_metrics(
            data_x=self.params_set["corr_metrics"].to_list(),
            data_y=self.params_set["MAPE_metrics"].to_list(),
            title="Metrics",
            x_label="corr_metrics",
            y_label="MAPE_metrics",
            save_path="img/metrics.jpeg"
        )

        # print(self.params_set.head(16))

        # logging.info(f"Inference will be done using {self.model_params['n_threads']} processes")
        # Theil_metrics = []
        # RMSPE_metrics = []
        # MAPE_metrics = []
        # plt.figure(figsize=(8, 8))
        # plt.title('Prediction traces')
        # for i in tqdm(range(self.params_set.shape[0])):
        #     Y_preds = []
        #     K_preds = []
        #     L_preds = []
        #     for t in range(self.data_params["end_year"] - self.data_params["start_year"]):
        #         if t == 0:
        #             L_t, K_t = self.model.L_0, self.model.Y_0/self.params_set["alpha_k"][i]
        #             Y_t = self.model.calc_gdp(
        #                 a=self.params_set["a"][i],
        #                 K=K_t,
        #                 L=L_t,
        #                 # L=self.dataset[cfg.LABOUR_COL][t],
        #                 gamma=self.params_set["gamma"][i],
        #                 alpha_k=self.params_set["alpha_k"][i]
        #             )
        #             Y_preds.append(Y_t)
        #             K_preds.append(K_t)
        #             L_preds.append(L_t)
        #         else:
        #             # L_t = self.model.calc_labor(self.params_set["n"][i], t)
        #             L_t = self.model.calc_labor(n, t)
        #             K_t = self.model.calc_capital(
        #                 s=self.params_set["s"][i],
        #                 Y=Y_preds[-1],
        #                 delta=self.params_set["delta"][i],
        #                 # n=self.params_set["n"][i],
        #                 n=n,
        #                 K=K_preds[-1]
        #             )
        #             Y_t = self.model.calc_gdp(
        #                 a=self.params_set["a"][i],
        #                 K=K_t,
        #                 # L=L_t,
        #                 L=self.dataset[cfg.LABOUR_COL][t],
        #                 gamma=self.params_set["gamma"][i],
        #                 alpha_k=self.params_set["alpha_k"][i]
        #             )
        #             Y_preds.append(Y_t)
        #             K_preds.append(K_t)
        #             L_preds.append(L_t)
        #     Y_preds = [el/1e9 for el in Y_preds]
        #     plt.plot(
        #         range(self.data_params["end_year"] - self.data_params["start_year"] - 7),
        #         Y_preds[:-7],
        #         color='orange',
        #         alpha=0.15
        #     )
        #     Theil_metrics.append(Theil(Y_preds, self.dataset[cfg.GDP_COL]))
        #     RMSPE_metrics.append(RMSPE(Y_preds, self.dataset[cfg.GDP_COL]))
        #     MAPE_metrics.append(MAPE(Y_preds, self.dataset[cfg.GDP_COL]))
        # plt.plot(
        #     range(self.data_params["end_year"] - self.data_params["start_year"] - 7),
        #     self.dataset[cfg.GDP_COL][:-7],
        #     color='red',
        #     alpha=0.85
        # )
        # plt.savefig("img/gdp_preds.jpeg")
        # plt.figure(figsize=(8, 8))
        # plt.title('Metrics')
        # plt.scatter(
        #     Theil_metrics,
        #     RMSPE_metrics
        # )
        # plt.savefig("img/metrics.jpeg")
        # self.params_set["Theil_metrics"] = Theil_metrics
        # self.params_set["MAPE_metrics"] = MAPE_metrics

        # logging.info("Calculated metrics for all combinations of parameters")

        # print(self.params_set.sort_values(
        #         by=["Theil_metrics", "MAPE_metrics"],
        #         ascending=[True, True]
        #     ).head(10)
        # )

        logging.info("Pipeline done!")


if __name__ == "__main__":
    fire.Fire(Pipeline)
