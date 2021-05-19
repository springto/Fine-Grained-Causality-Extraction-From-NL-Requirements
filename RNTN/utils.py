import json
import logging
import time
from datetime import datetime

TRAINING_INFO_LEVEL_NUM = 15

class Params:
    # Source code from <https://github.com/cs230-stanford/cs230-code-examples>
    """Class that loads training params and model hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def now_as_str_f():
    return "{:%Y_%m_%d---%H_%M_%f}".format(datetime.now())


def get_logger(log_path=None, log_level=logging.DEBUG):
    logging.getLogger().setLevel(logging.WARNING)
    logger = logging.getLogger('uncertainty_estimation_in_dl')

    if log_path and not logger.handlers:
        # set the level
        logger.setLevel(log_level)

        # Logging to a file
        f = '[%(asctime)s][%(levelname)s][%(message)s]'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(stream_handler)

        # Add a new level between debug and info for printing logs while training
        logging.addLevelName(TRAINING_INFO_LEVEL_NUM, 'TINFO')
        setattr(logger, 'tinfo', lambda *args: logger.log(TRAINING_INFO_LEVEL_NUM, *args))

    return logger
