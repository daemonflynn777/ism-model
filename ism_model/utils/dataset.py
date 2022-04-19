from dataclasses import dataclass
import logging
import pandas as pd


@dataclass
class Data():
    source_file: str = None
    dest_file: str = None

    def __post_init__(self):
        if self.source_file:
            logging.info(f"Data will be loaded from {self.source_file}")
        elif self.dest_file:
            logging.info(f"Data will be saved to {self.dest_file}")
        else:
            raise ValueError("No source_file or dest_file was provided")

        self.source_file_format = self.source_file.rsplit(".", 1)[1]
        self.dest_file_format = self.dest_file.rsplit(".", 1)[1]

    def load(self):
        if self.source_file_format == "csv":
            return pd.read_csv(self.source_file)
        elif self.source_file_format == "xls":
            return pd.read_excel(self.source_file)
        else:
            raise ValueError("Source file has invalid type. Supported types are .csv and .xls")
