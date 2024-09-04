import json
import logging

import daiquiri
import fire
from marshmallow import RAISE, Schema, fields, pre_load

from clb.classify.classify import get_parser
from clb.classify.classify import main as classify_
from clb.virtules.utils import reformat_to_parser, set_path_with_env


logger = daiquiri.getLogger(__name__)


class ParametersSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    segment_dir = fields.String(required=True)
    channel = fields.List(fields.Integer(required=True), required=True)
    start = fields.Integer(required=False, allow_none=True)
    stop = fields.Integer(required=False, allow_none=True)

    @pre_load
    def remove_start_stop_if_none(self, data, **kwargs):
        return {k: v for k, v in data.items() if not (k in ("start", "stop") and (v == None))}


def main(parameters):
    """
    Run classification with specified `parameters`. 
    Prerequirements: environment variable set - AZ_BATCH_JOB_PREP_WORKING_DIR

    Args:
        parameters (str): Path to a json file with following parameters:
                          input (str): path to an input file
                          output (str): path to the output dir
                          segment_dir (str): path to the segmentation dir
                          channel (list): which channel of the INPUT_FILE is cd8
                          start (int): Slice number to start from (optional)
                          stop (int): Slice number to stop to (optional)
    """
    logger.info("Loading classification information from: {}".format(parameters))

    parser = get_parser()
    schema = ParametersSchema(unknown=RAISE)

    with open(parameters, mode="r") as fp:
        params = json.load(fp)
    validated = schema.load(params)
    tweaked = tweak_params(validated)
    parser_params = reformat_to_parser(tweaked)
    args = parser.parse_args(parser_params)

    logger.info("Calling with: {}".format(args))
    classify_(args)


def tweak_params(parameters):
    parameters["input"] = set_path_with_env(parameters["input"])
    parameters["outputs"] = [
        set_path_with_env(parameters["output"], "{name}.ims"),
        set_path_with_env(parameters["output"], "series0"),
        set_path_with_env(parameters.pop("output"), "classification.tif"),
    ]
    parameters["labels"] = set_path_with_env(parameters.pop("segment_dir"), "labels.tif")
    parameters["channels"] = ",".join(map(str, parameters.pop("channel")))
    parameters["channel_name"] = "cd8"
    parameters["channel_color"] = "red"
    parameters["discrete"] = "binary"
    parameters["uff_colors"] = ("gray", "red")
    parameters["model"] = "models/classification/model_7_eager_torvalds_2_perc_good_class_cd8.pkl"
    return parameters


if __name__ == "__main__":
    fire.Fire(main)
