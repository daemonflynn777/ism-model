import fire
import os
import logging
import logging.config
from multiprocessing import Pool
import pandas as pd

from ism_model.utils.yaml import load_yaml_safe
from ism_model.utils.dataset import Data
import ism_model.config as cfg
from ism_model.model.ism import Model


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
    def inference(df: pd.Dataframe):
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
        print(self.dataset)

        logging.info("Initializing the model")
        self.model = Model(L_0=self.dataset[cfg.LABOUR_COL][0])

        logging.info("Creating mesh grid for model's params")
        self.model.create_params_set(
            coeffs=self.model_params["coeffs"],
            n_splits=self.model_params["n_splits"]
        )

        logging.info(f"Inference will be done using {self.model_params['n_threads']} processes")


if __name__ == "__main__":
    fire.Fire(Pipeline)
