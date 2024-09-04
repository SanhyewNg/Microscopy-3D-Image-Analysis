import json
import logging

import daiquiri
import fire
from marshmallow import RAISE, Schema, fields, post_load

from clb.dataprep.imaris.export_imaris import get_parser
from clb.dataprep.imaris.export_imaris import main as export
from clb.virtules.utils import get_env_variable, reformat_to_parser, set_path_with_env


logger = daiquiri.getLogger(__name__)


class ChannelSchama(Schema):
    series_path = fields.String(required=True)
    channel_name = fields.String(required=True)
    channel = fields.Integer(required=True)
    color = fields.String(required=False, allow_none=True)


class ParametersSchema(Schema):
    input = fields.List(fields.Nested(ChannelSchama), required=True)
    output = fields.String(required=True)


def main(parameters):
    """
    Run export_imaris with specified `parameters`. 
    Prerequirements: environment variable set - AZ_BATCH_JOB_PREP_WORKING_DIR
    
    Args:
        parameters (str): Path to a json file with following parameters:
                          input (list): each input channel (dict) should be described:
                            series_path: str,
                            channel_name: str,
                            channel: int.
                            color: None | str. `color` is optional
                          output (str): path to the output file
    """
    logger.info("Loading params from: {}".format(parameters))
    parser = get_parser()
    schema = ParametersSchema(unknown=RAISE)

    with open(parameters, mode="r") as fp:
        params = json.load(fp)
    validated = schema.load(params)
    tweaked = tweak_params(validated)
    parser_params = reformat_to_parser(tweaked)
    args = parser.parse_args(parser_params)

    logger.info("Calling with: {}".format(vars(args)))
    export(**vars(args))


def stringify_channel_input(channel_input):
    channel_argument = f"{channel_input['series_path']},{channel_input['channel_name']},{channel_input['channel']}"
    if channel_input.get("color") is not None:
        channel_argument += f',{channel_input["color"]}'
    logger.info(channel_argument)
    return channel_argument


def tweak_params(parameters):
    parameters["inputs"] = [
        set_path_with_env(stringify_channel_input(input_param)) for input_param in parameters.pop("input")
    ]
    parameters["output_path"] = set_path_with_env(parameters.pop("output"))
    return parameters


if __name__ == "__main__":
    fire.Fire(main)
