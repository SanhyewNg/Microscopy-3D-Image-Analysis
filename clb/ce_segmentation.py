import argparse
import glob
import os
import shutil
import subprocess
import yaml
import daiquiri
import logging

from clb.evaluate.evaluator_segment import main as run_evaluator, parse_arguments as parse_evaluator
from clb.run import main as run_run, get_parser

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def call_evaluator(output, model, data, datasets, skip_details):
    """Create an evaluator call based on parameters.

    Args:
        output: path to output directory
        model: path to h5 model file
        data: path to directory with train, val and test dirs
        datasets: datasets to be used: e.g. "train+val", "test"
        skip_details: do not create segmentation details

    Raises:
        RuntimeError: when process that is called in `subprocess.check_call`
        encounters any error and exits incorrectly, specific exception
        `CalledProcessError` is raised by `subprocess.check_call`. It is
        caught here and proper information is printed.

    """
    logger.info("Running evaluator using {}".format(model))

    model_name = os.path.splitext(os.path.split(model)[1])[0]
    probs = os.path.join(output, model_name + '_probs')
    labels = os.path.join(output, model_name + '_labels')

    params = [
        "--name", model_name,
        "--data", data,
        "--datasets", datasets,
        "--probs", probs,
        "--labels", labels,
        "--output", os.path.join(output, "results"),
        "--model", os.path.join("models", model)]
    if skip_details:
        params += ["--skip_details"]

    try:
        run_evaluator(parse_evaluator(params))
    except Exception:
        raise RuntimeError('clb.evaluator for model {} failed.'.format(model))


def run_evaluate_compare(args, models_to_evaluate):
    logger.info("Running evaluate and compare on {} models".format(len(models_to_evaluate)))
    for model in models_to_evaluate:
        call_evaluator(args.output, model, args.data, args.datasets, args.skip_details)

    compare_call = 'python vendor/ep/compare.py ' + \
                   os.path.join(args.output, 'results')

    for model in models_to_evaluate:
        compare_call += ' ' + os.path.splitext(model)[0] + '_labels/Output'

    subprocess.run(compare_call, stderr=subprocess.STDOUT, shell=True, check=True,
                   bufsize=0)


def run_solution(output, model, data):
    """
    Run entire solution using provided model on a sample image.
    Args:
        output: path where to store results
        model: model to use
        data: root folder for input imagery
    """
    all_tiff = sorted(glob.iglob(os.path.join(data, '**/*.tif'), recursive=True))

    test_file_prefix = "segment_test"
    test_file_input = test_file_prefix + "_input.tif"
    test_file_output = test_file_prefix + "_output.tif"

    tiff_to_use = all_tiff[0]
    logger.info("Running instance segmentation on {0} using model {1}".format(tiff_to_use, model))

    shutil.copy(tiff_to_use, os.path.join(output, "results", test_file_input))
    args = [
        '--input', os.path.join(output, "results", test_file_input),
        '--outputs', os.path.join(output, "results", test_file_output),
        '--model', os.path.join("models", model)
    ]
    parser = get_parser()
    args = parser.parse_args(args)

    run_run(args)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare multiple segmentation models.',
                                     add_help=False)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--data', help='path to directory with train, val and test dirs',
                          required=True)
    required.add_argument('--datasets', help='datasets to be used: e.g. "train+val", "test"',
                          required=True)
    required.add_argument('--output', help='output directory for results',
                          required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--skip_details', dest='skip_details', action='store_true',
                          help='skip segmentation details')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    with open('models/models_to_evaluate.yaml', 'r') as f:
        models_to_evaluate = yaml.load(f)

    run_evaluate_compare(args, models_to_evaluate)
    run_solution(args.output, models_to_evaluate[-1], args.data)
