import os
import yaml

YEAR_COL = "Year"
GDP_COL = "Gross capital formation"
LABOUR_COL = "Population"
FUND_COL = ""
CONSUMPTION_COL = "Final consumption expenditure"

LOG_DIR = "logs"
LOG_NAME = "pipeline.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_NAME)

LOG_YAML_CONFIG = f"""
version: 1
disable_existing_loggers: True
formatters:
  simple:
    format: '[%(asctime)s]–[%(levelname)s]–[%(message)s](%(filename)s:%(lineno)s)'
    datefmt: "%Y-%m-%d %H:%M:%S"
handlers:
  console:
    level: INFO
    formatter: simple
    class: logging.StreamHandler
    stream: ext://sys.stdout
  flat_file:
    level: INFO
    formatter: simple
    class: logging.FileHandler
    filename: {LOG_PATH}
    mode: w
loggers:
  '':
    level: INFO
    handlers: [console, flat_file]
    propagate: no
  alembic.runtime.migration:
    level: ERROR
    handlers: [console, flat_file]
"""

LOG_DICT_CONFIG = yaml.safe_load(LOG_YAML_CONFIG)
