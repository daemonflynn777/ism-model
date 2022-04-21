import fire
import os
import logging
import logging.config

from ism_model.utils.yaml import load_yaml_safe
from ism_model.utils.dataset import Data
import ism_model.config as cfg


class Pipeline:
    def __init__(self, run_config: str):
        logging.info("Started the pipeline")
        logging.config.dictConfig(cfg.LOG_DICT_CONFIG)
        self.run_config = run_config
        self.params = load_yaml_safe(yaml_path=self.run_config)

        self.tech_params = self.params["tech"]
        self.data_params = self.params["data"]

    def load_dataset(self):
        self.dataset = Data(
            source_file=os.path.join(
                self.tech_params["data_dir"],
                self.data_params["file"]
            )
        ).load()

    def run(self):
        self.load_dataset()
        print(self.dataset.head())


if __name__ == "__main__":
    fire.Fire(Pipeline)
