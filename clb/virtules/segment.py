import json
import logging

import daiquiri
import fire
from marshmallow import RAISE, Schema, fields, pre_load

from clb.run import get_parser
from clb.run import main as segment
from clb.virtules.utils import set_path_with_env, reformat_to_parser


logger = daiquiri.getLogger(__name__)


class ParametersSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    channel = fields.Integer(required=True, allow_none=False)
    start = fields.Integer(required=False, allow_none=False)
    stop = fields.Integer(required=False, allow_none=False)

    @pre_load
    def remove_start_stop_if_none(self, data, **kwargs):
        return {k: v for k, v in data.items() if not (k in ("start", "stop") and (v == None))}


def main(parameters):
    """
    Run segmentation with specified `parameters`. 
    Prerequirements: environment variable set - AZ_BATCH_JOB_PREP_WORKING_DIR
    
    Args:
        parameters (str): Path to a json file with following parameters:
                          input (str): path to an input file
                          output (str): output dir
                          channel (int): Number of channel to use in segmentation
                          start (int): Slice number to start from (optional)
                          stop (int): Slice number to stop to (optional)
    """
    logger.info("Loading segmentation information from: {}".format(parameters))
    parser = get_parser()
    schema = ParametersSchema(unknown=RAISE)

    with open(parameters, mode="r") as fp:
        params = json.load(fp)
    validated = schema.load(params)
    tweaked = tweak_params(validated)
    parser_params = reformat_to_parser(tweaked)
    args = parser.parse_args(parser_params)

    logger.info("Calling with: {}".format(args))
    segment(args)


def tweak_params(parameters):
    parameters["input"] = set_path_with_env(parameters["input"])
    parameters["outputs"] = [
        set_path_with_env(parameters["output"], "labels.tif"),
        set_path_with_env(parameters["output"], "series0"),
        set_path_with_env(parameters.pop("output"), "{name}.ims"),
    ]
    parameters["use_channel"] = parameters.pop("channel")
    parameters["model"] = "models/model_8_angry_ptolemy.h5"
    return parameters


if __name__ == "__main__":
    fire.Fire(main)
