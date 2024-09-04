import json
import logging

import daiquiri
import fire
from marshmallow import RAISE, Schema, fields, pre_load

from clb.run import get_parser
from clb.stats.spatial_stats import main as stats
from clb.virtules.utils import set_path_with_env


logger = daiquiri.getLogger(__name__)


class ParametersSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    segment_dir = fields.String(required=True)
    ref_plot_name = fields.String(required=True)
    ref_class_path = fields.String(required=True)
    tested_classes_names = fields.List(fields.String(), required=True)
    tested_classes_paths = fields.List(fields.String(), required=True)
    filter_double_positives = fields.Boolean(required=False, allow_none=True)

    @pre_load
    def remove_start_stop_if_none(self, data, **kwargs):
        return {k: v for k, v in data.items() if not (k in ("start", "stop") and (v == None))}


def main(parameters):
    """
    Run spatial_stats with specified `parameters`. 

    Prerequirements: environment variable set - AZ_BATCH_JOB_PREP_WORKING_DIR
    
    Args:
        parameters (str): Path to a json file with following parameters:
                          input (str): path to an input file
                          output (str): directory to save results
                          segment_dir (str): path to the segmentation dir
                          ref_plot_name (str): name of the reference class plot
                          ref_class_path (str): path to the classification results of reference class
                          tested_classes_names (list): legend entries for test classes
                          tested_classes_paths (list): paths to test classes
                          filter_double_positives (bool): should double positives be filtered from tested classes (optional)
    """
    logger.info("Loading stats information from: {}".format(parameters))
    schema = ParametersSchema(unknown=RAISE)

    with open(parameters, mode="r") as fp:
        params = json.load(fp)
    validated = schema.load(params)
    tweaked = tweak_params(validated)

    logger.info(f"Calling with: {tweaked}")
    stats(**tweaked)


def tweak_params(parameters):
    parameters["input"] = set_path_with_env(parameters["input"])
    parameters["output_graph"] = set_path_with_env(parameters["output"], "graph.png")
    parameters["output_data_dir"] = set_path_with_env(parameters.pop("output"), "data")
    parameters["labels"] = set_path_with_env(parameters.pop("segment_dir"), "labels.tif")
    parameters["ref_class_path"] = set_path_with_env(parameters["ref_class_path"])
    parameters["tested_classes_paths"] = [set_path_with_env(path) for path in parameters["tested_classes_paths"]]
    return parameters


if __name__ == "__main__":
    fire.Fire(main)
