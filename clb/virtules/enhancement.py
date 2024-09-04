import json
from typing import Any, Dict, List, Tuple, Optional
from imageio import volread

import daiquiri
import fire
from marshmallow import INCLUDE, RAISE, Schema, ValidationError, fields, post_load, validates, validates_schema

from clb.denoising.utils import select_stride
from clb.denoising.denoise import denoise
from clb.virtules.utils import set_path_with_env
from clb.dataprep.readers import get_volume_reader


logger = daiquiri.getLogger(__name__)


class DenoiseSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    channel = fields.List(fields.Integer(required=True, allow_none=False), required=True)


class EnhancementConfigSchema(Schema):
    name = fields.String(required=True)

    class Meta:
        unknown = INCLUDE


class ParametersSchema(Schema):
    input = fields.String(required=True)
    output = fields.String(required=True)
    enhancements = fields.List(fields.Nested(EnhancementConfigSchema), required=True, allow_none=False)

# TODO: server side of enhancements is not implemented !
def main(parameters: str):
    """Run enhancement on file specified in `parameters`.
    Multiple enhancements can be applied in one step. Parameters of each of them are stored in
    `enhancements` list as json. Each enhancement parameters are validated separately.

    Args:
        parameters (str): Path to a json file with following parameters:
                          input (str): path to an input file
                          output (str): path to an output file
                          enhancements (list): configuration of enhancements
    """
    logger.info("Loading enhancement parameters from: {}".format(parameters))

    schema = ParametersSchema(unknown=RAISE)
    with open(parameters, mode="r") as fp:
        params = json.load(fp)

    validated = schema.load(params)

    for enhancement_config in validated["enhancements"]:
        enhancement_type = enhancement_config.pop("name")
        enhancement_schema = ENHANCEMENT_VALIDATOR_MAPPING[enhancement_type](unknown=RAISE)

        enhancement_config["input"] = validated["input"]
        enhancement_config["output"] = validated["output"]

        parameters = enhancement_schema.load(enhancement_config)
        tweaked = TWEAK_MAPPING[enhancement_type](parameters)

        logger.info(f"Calling {enhancement_type} with parametes: {tweaked}")
        ENHANCEMENT_MAPPING[enhancement_type](**tweaked)


def denoise_tweak(parameters):
    input_ = set_path_with_env(parameters.pop("input"))
    output_ = set_path_with_env(parameters.pop("output"))
    image_shape_2d = get_image_shape(input_)
    parameters.update(set_stride_and_patch_size(image_shape_2d))
    parameters["channel"] = parameters["channel"][0]
    parameters["input"] = input_
    parameters["output"] = output_
    parameters["model"] = "models/denoising/model0.h5"
    return parameters


def set_stride_and_patch_size(
    image_shape_2d: Tuple[int, int],
    patch_size: Tuple[int, int] = (256, 256),
    patch_stride: Optional[Tuple[int, int]] = None,
):
    return {
        "patches_shape": (256, 256),
        "patches_stride": patch_stride or select_stride(image_shape_2d, patch_size=patch_size),
    }


# TODO: get image shape can be improved to use readers - `reader.shape` now it could not handle properly all shape cases
def get_image_shape(image_path: str) -> Tuple[int, int]:
    image = volread(image_path)
    if image.ndim == 3:
        return image.shape[:2]
    if image.ndim == 4:
        return image.shape[1:3]
    else:
        raise ValueError("Could not determine input image shape")


ENHANCEMENT_MAPPING = {"denoise": denoise}
ENHANCEMENT_VALIDATOR_MAPPING = {"denoise": DenoiseSchema}
TWEAK_MAPPING = {"denoise": denoise_tweak}


if __name__ == "__main__":
    fire.Fire(main)
