import inspect
import json
import logging
from copy import deepcopy

import daiquiri
import fire
from marshmallow import RAISE, Schema, fields, pre_load

from clb.run import get_parser
from clb.stats.all_stats import main as stats
from clb.stats.scatterplots import plot_scatterplots
from clb.virtules.utils import set_path_with_env


logger = daiquiri.getLogger(__name__)

STATS_ARGS = list(inspect.signature(stats).parameters.keys())
PLOT_ARGS = list(inspect.signature(plot_scatterplots).parameters.keys())


class ParametersSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    segment_dir = fields.String(required=True)
    channels = fields.List(fields.Integer(), required=True)
    channel_names = fields.List(fields.String(), required=True)
    start = fields.Integer(required=False, allow_none=True)
    stop = fields.Integer(required=False, allow_none=True)
    classes = fields.Dict(required=True)

    @pre_load
    def remove_start_stop_if_none(self, data, **kwargs):
        return {k: v for k, v in data.items() if not (k in ("start", "stop") and (v == None))}


def main(parameters):
    """
    Run all_stats with specified `parameters`. 

    Prerequirements: environment variable set - AZ_BATCH_JOB_PREP_WORKING_DIR
    
    Args:
        parameters (str): Path to a json file with following parameters:
                          input (str): path to an input file
                          output (str): directory to save results
                          segment_dir (str): path to the segmentation dir
                          channels (str): comma separated list of channels to use
                          channel_names (str): comma separated list of channels names
                          classes (dict): {"PanCK": <path_panck>, ...}
                          start (int): Slice number to start from (optional)
                          stop (int): Slice number to stop to (optional)
    """
    logger.info("Loading stats information from: {}".format(parameters))
    schema = ParametersSchema(unknown=RAISE)

    with open(parameters, mode="r") as fp:
        params = json.load(fp)
    validated = schema.load(params)
    stats_params = get_stats_params(validated)
    plot_params = get_plot_params(validated)

    logger.info("Calling stats_params with: {}".format(stats_params))
    logger.info("Calling plot_params with: {}".format(plot_params))
    stats(**stats_params)
    plot_scatterplots(**plot_params)


def get_stats_params(params):
    parameters = deepcopy(params)
    parameters["input"] = set_path_with_env(parameters["input"])
    parameters["output"] = set_path_with_env(parameters["output"], "{name}_")
    parameters["labels"] = set_path_with_env(parameters["segment_dir"], "labels.tif")
    parameters = {k: v for k, v in parameters.items() if k in STATS_ARGS}
    parameters.update(parameters.pop("classes"))
    return parameters


def get_plot_params(params):
    parameters = deepcopy(params)
    parameters["input"] = set_path_with_env(parameters["input"])
    parameters["stats_path"] = set_path_with_env(parameters["output"], "{name}_nuclei_stats.csv")
    parameters["output_dir"] = set_path_with_env(parameters["output"], "scatterplots")
    parameters = {k: v for k, v in parameters.items() if k in PLOT_ARGS}
    return parameters


if __name__ == "__main__":
    fire.Fire(main)
