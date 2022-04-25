import logging
import logging.config
import fire
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import model_selection

from ism_model import config as cfg
from ism_model.utils.dataset import Data


class Model():
    def __init__(self):
        logging.info("Started the ISM model ...")

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

    def load_data(self):
        self.dataset = Data.load(source_file="")
        pass

    def save_data(self):
        pass

    def compute_metrics(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    fire.Fire(Model)
