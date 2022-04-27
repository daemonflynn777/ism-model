import fire
import os
import logging
import logging.config
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

from ism_model.utils.yaml import load_yaml_safe
from ism_model.utils.dataset import Data
import ism_model.config as cfg
from ism_model.model.ism import Model
from ism_model.metrics.metrics import Theil, RMSPE


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
        return Data(
            source_file=os.path.join(
                self.tech_params["data_dir"],
                self.data_params["file"]
            )
        ).load()

    @staticmethod
    def inference(df: pd.DataFrame):
        pass

    def run(self):
        # for el in self.model_params["coeffs"]:
        #     print(el)
        # self.model.create_params_set(self.model_params["coeffs"], self.model_params["n_splits"])
        logging.info("Loading train data")
        self.dataset = self.load_dataset()
        self.dataset = self.dataset[
            (self.dataset[cfg.YEAR_COL] >= self.data_params["start_year"]) &
            (self.dataset[cfg.YEAR_COL] <= self.data_params["end_year"])
        ]
        self.dataset = self.dataset[[
            cfg.GDP_COL,
            cfg.LABOUR_COL
        ]].reset_index(drop=True)
        # print(self.dataset)

        logging.info("Initializing the model")
        self.model = Model(L_0=self.dataset[cfg.LABOUR_COL][0], Y_0=self.dataset[cfg.GDP_COL][0])

        logging.info("Creating mesh grid for model's params")
        self.params_set = self.model.create_params_set(
            coeffs=self.model_params["coeffs"],
            n_splits=self.model_params["n_splits"]
        )
        # print(self.params_set.head(16))

        logging.info(f"Inference will be done using {self.model_params['n_threads']} processes")
        Theil_metrics = []
        RMSPE_metrics = []
        for i in tqdm(range(self.params_set.shape[0])):
            Y_preds = []
            K_preds = []
            L_preds = []
            for t in range(self.data_params["end_year"] - self.data_params["start_year"]):
                if t == 0:
                    L_t, K_t = self.model.L_0, self.model.Y_0/self.params_set["alpha_k"][i]
                    Y_t = self.model.calc_gdp(
                        a=self.params_set["a"][i],
                        K=K_t,
                        L=L_t,
                        gamma=self.params_set["gamma"][i]
                    )
                    Y_preds.append(Y_t)
                    K_preds.append(K_t)
                    L_preds.append(L_t)
                else:
                    L_t = self.model.calc_labor(self.params_set["n"][i], t)
                    K_t = self.model.calc_capital(
                        s=self.params_set["s"][i],
                        Y=Y_preds[-1],
                        delta=self.params_set["delta"][i],
                        n=self.params_set["n"][i],
                        K=K_preds[-1]
                    )
                    Y_t = self.model.calc_gdp(
                        a=self.params_set["a"][i],
                        K=K_t,
                        L=L_t,
                        gamma=self.params_set["gamma"][i]
                    )
                    Y_preds.append(Y_t)
                    K_preds.append(K_t)
                    L_preds.append(L_t)
            Theil_metrics.append(Theil(Y_preds, self.dataset[cfg.GDP_COL]))
            RMSPE_metrics.append(RMSPE(Y_preds, self.dataset[cfg.GDP_COL]))
        self.params_set["Theil_metrics"] = Theil_metrics
        self.params_set["RMSPE_metrics"] = RMSPE_metrics

        logging.info("Calculated metrics for all combinations of parameters")

        print(self.params_set.sort_values(
                by=["Theil_metrics", "RMSPE_metrics"],
                ascending=[True, True]
            ).head(10)
        )

        logging.info("Pipeline done!")


if __name__ == "__main__":
    fire.Fire(Pipeline)
