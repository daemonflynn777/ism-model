import logging
import logging.config
from syslog import LOG_LOCAL0
import fire
from typing import Dict
import pandas as pd
import numpy as np
import math

from ism_model import config as cfg
from ism_model.utils.dataset import Data


class Model():
    def __init__(self, L_0: float):
        logging.info("Started the ISM model ...")

        self.L_0 = L_0

        self.a = 0.0
        self.gamma = 0.0
        self.alpha_k = 0.0
        self.n = 0.0
        self.s = 0.0
        self.delta = 0.0

    @staticmethod
    def create_params_set(coeffs: Dict[str, str], n_splits: int) -> pd.DataFrame:
        parsed_coeffs = {
            param: np.linspace(
                start=float(coeffs[param].split(";")[0]),
                stop=float(coeffs[param].split(";")[1]),
                num=n_splits
            ) for param in coeffs
        }
        grid_df = pd.DataFrame(
            data=np.array(np.meshgrid(*parsed_coeffs.values())).reshape(-1, len(parsed_coeffs)),
            columns=parsed_coeffs.keys()
        )
        return grid_df

    @staticmethod
    def calc_gdp(a: float, K: float, L: float, gamma: float) -> float:
        return (a*(K**gamma) + (1-a)*(L**gamma))**(1/gamma)

    def calc_labor(self, n: float, t: int) -> int:
        return self.L_0*math.exp(n*t)

    def calc_capital(self, s: float, Y: float, delta: float, n: float, K: float) -> float:
        return s*Y + (1 - delta - n)*K

    def load_data(self, source_file: str):
        self.dataset = Data(source_file=source_file).load()
        self.dataset = self.dataset[[
            cfg.GDP_COL,
            cfg.LABOUR_COL
        ]]

    def save_data(self):
        pass

    def compute_metrics(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    fire.Fire(Model)
