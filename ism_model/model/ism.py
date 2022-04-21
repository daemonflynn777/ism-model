import logging
import logging.config
import fire

from ism_model import config as cfg
from ism_model.utils import Data


class Model():
    def __init__(self, run_config: str):
        logging.info("Started the ISM model ...")
        pass

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
