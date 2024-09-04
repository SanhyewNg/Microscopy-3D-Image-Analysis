import logging
import os
from pathlib import Path

import daiquiri


logger = daiquiri.getLogger(__name__)


class MissingEnvException(Exception):
    """Missing Environment Variable"""

    def __init__(self, message):
        self.message = message
        logger.exception(message)


def reformat_to_parser(params):
    """Reformat params to fit argpars parser schema input.

    Args:
        params (dict): Dict with parametes.
    
    Return:
        result (list): Argparse format list: ["--input", "input_value", ...]
    """
    result = []
    listed_params = []
    for param_name, param in params.items():
        listed_params.append("--{}".format(param_name))
        listed_params.append(param)
    for param in listed_params:
        if isinstance(param, (list, tuple)):
            result.extend(param)
        else:
            result.append(param)
    return list(map(stringify_number, result))


def get_env_variable(variable_name):
    try:
        env_var = os.environ["{}".format(variable_name)]
    except KeyError:
        raise MissingEnvException("Environment variable: {} not found".format(variable_name))
    return env_var


def stringify_number(param):
    if isinstance(param, (int, float)):
        param = "{}".format(param)
    return param


def set_path_with_env(base_dir, suffix_dir=""):
    """Set dir based on `env` settings. If require append path"""
    az_work_dir = get_env_variable("AZ_BATCH_JOB_PREP_WORKING_DIR")
    with_env = base_dir.format(AZ_BATCH_JOB_PREP_WORKING_DIR=az_work_dir)
    return str(Path(with_env, suffix_dir))
