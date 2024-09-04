import argparse
import csv
import distutils.dir_util as dir_util
import itertools
import os
from random import shuffle
import sys
import shutil
import tempfile
from collections import OrderedDict
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from clb.evaluate.evaluator_segment import main as evaluator_main
from clb.yaml_utils import save_args_to_yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tune postprocess hyperparameters using train and validation sets.',
                                     add_help=False)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--model', help='path to h5 model along with training yaml params', required=True)
    required.add_argument('--method', help='method used for instance segmentation: components or watershed',
                          required=True)
    required.add_argument('--data',
                          help='path to directory with train, val and test dirs, '
                               'required if --annotated_input or --annotated_gt not specified', required = True)
    required.add_argument('--output', help='where to store all the evaluation', required=True)

    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--datasets',
                          help='used and required if --data is specified, describes which datasets should be used:'
                               'e.g. "T8/train+T8/val", "test", "T8/val+T8/test"',
                          default='T8/val')
    optional.add_argument('--eval_only', type=int, help='number of param sets to evaluate', default=1000000)
    optional.add_argument('--workers', type=int, help='number of workers used in parameter search', default=1)
    optional.add_argument('--temp_results', help='where to store calculation, if not provided temporary folder is used')
    optional.add_argument('--report_all', dest='report_all', action='store_true',
                          help='report all results existing results regardless of currently specified')
    optional.add_argument('--shuffle', dest='shuffle', action='store_true',
                          help='shuffle parameter sets to test')
    optional.add_argument('--draw_details', dest='draw_details', action='store_true',
                          help='prepare segmentation details')
    optional.add_argument('--discard_images', dest='discard_images', action='store_true',
                          help='remove all images produced for evaluation')
    optional.add_argument('--seed', type=int, help='seed for generating ' +
                                                   'random hyperparameters', default=48)
    parser.set_defaults(draw_details=False)
    parser.set_defaults(shuffle=False)
    parser.set_defaults(discard_images=False)
    parser.set_defaults(report_all=False)
    return parser.parse_args()


def run_comparison(output_dir, list_of_label_folders):
    if list_of_label_folders is None:
        list_of_label_folders = os.listdir(output_dir)

    output_dirs_for_solutions = [os.path.join(f, "Output") for f in list_of_label_folders]
    existing_output_dirs = [out_dir for out_dir in output_dirs_for_solutions if os.path.isdir(os.path.join(output_dir, out_dir))]
    sorted_output_dirs = sorted(existing_output_dirs)

    temp_file = None
    try:
        # If there are very many solutions their names have to go through temp file.
        if len(sorted_output_dirs) > 50:
            temp_file = "temp_solutions_to_compare.txt"
            with open(temp_file, "w") as f:
                f.writelines("\n".join(sorted_output_dirs))
            solutions_param = temp_file
        else:
            solutions_param = " ".join(sorted_output_dirs)

        ep_compare_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "vendor", "ep",
                                       "compare.py")

        final_call = ep_compare_path + " " + output_dir + " " + solutions_param
        print(final_call)

        os.system('python3 ' + final_call)
    finally:
        if temp_file is not None:
            os.remove(temp_file)


class HypertuneParams:
    def __init__(self, basic_params_dict, params_dict_list):
        self.basic_params = OrderedDict(basic_params_dict)
        self.all_params = HypertuneParams.product_with_names(params_dict_list)
        self.abbrevs = {}

    @staticmethod
    def product_with_names(dict_with_values):
        return [OrderedDict(zip(dict_with_values.keys(), x)) for x in itertools.product(*dict_with_values.values())]

    def format_value(self, val):
        if isinstance(val, float):
            return "{0:g}".format(val)
        else:
            return str(val)

    def get_abbreviation(self, param):
        return self.abbrevs.get(param) or param[:5]

    def get_file_prefix(self, param_set):
        params_list = ["{0}_{1}".format(self.get_abbreviation(param), self.format_value(value))
                       for param, value in param_set.items() if param not in self.abbrevs or self.abbrevs[param] is not None]
        return "_".join(params_list)

    def get_description(self, param_set):
        params_list = ["{0}={1}".format(param, self.format_value(value)) for param, value in param_set.items()]
        return ", ".join(params_list)

    def get_all_complete_params_sets(self):
        for d in self.all_params:
            complete_set = self.basic_params.copy()
            complete_set.update(d)
            yield complete_set


def run_one_parameter_set(description_evaluator_args):
    """
    Helper function for multiprocessing
    Args:
        description_evaluator_args: tuple of description and parameters args
    """
    description, evaluator_args = description_evaluator_args

    print("Running for {0}".format(description))
    evaluator_main(evaluator_args)
    return evaluator_args.name + "_labels"


def tune(args, hypertune_params):
    """
    This method runs evaluator on a set of parameters fo identify.
    It uses only train and validation sets for that purpose.
    
    The evaluation of a single set is then compared using EP.
    """
    np.random.seed(args.seed)

    temp_dir = None
    try:
        # Make temporary folder for intermediate probability volume.
        temp_dir = tempfile.mkdtemp()
        if args.temp_results is not None:
            os.makedirs(args.temp_results, exist_ok=True)
            root_results_dir = args.temp_results
        else:
            root_results_dir = temp_dir

        os.makedirs(args.output, exist_ok=True)
        probs_path = os.path.join(root_results_dir, "probs")

        produced_result_dirnames = []

        def should_skip(hyper_params, params_set):
            params_name = hyper_params.get_file_prefix(params_set)
            return os.path.isdir(os.path.join(args.output, params_name + "_labels", 'Output'))

        # Get all possible sets of params.
        all_params_sets = list(hypertune_params.get_all_complete_params_sets())
        if args.shuffle:
            shuffle(all_params_sets)

        # Gather previously evaluated results.
        params_sets_already_evaluated = [s for s in all_params_sets if should_skip(hypertune_params, s)]
        names_of_already_evaluated = [hypertune_params.get_file_prefix(ps) for ps in params_sets_already_evaluated]
        folder_of_already_evaluated = [name + "_labels" for name in names_of_already_evaluated]
        produced_result_dirnames += folder_of_already_evaluated

        # List of parameters for which we actually run evalution.
        params_sets_to_run = [s for s in all_params_sets if not should_skip(hypertune_params, s)][:args.eval_only]

        def prepare_evaluator_args(hypertune_params, params_set):
            """
            Prepare Namespace object with parameters that evaluator can be run on.
            It dumps the parameters for instance segmentation to yaml file.

            Args:
                hypertune_params: HypertuneParams object used to prepare params_set
                params_set: dictionary with params to evalute

            Returns:
                Namespace object ready to use by evaluator.
            """
            name = hypertune_params.get_file_prefix(params_set)
            identify_yaml_temp = os.path.join(args.output, name)
            labels_path = os.path.join(root_results_dir, name)

            evaluator_args = argparse.Namespace()

            evaluator_args.data = args.data
            evaluator_args.datasets = args.datasets
            evaluator_args.probs = probs_path
            evaluator_args.output = args.output
            evaluator_args.skip_details = not args.draw_details
            evaluator_args.identify_yaml = identify_yaml_temp + ".yaml"

            evaluator_args.model = args.model
            evaluator_args.regen_prob = False
            evaluator_args.regen_labels = False
            evaluator_args.labels = labels_path
            evaluator_args.name = name
            evaluator_args.minimum_blob_overlap = 0.6

            evaluator_args.discard_probs = args.discard_images
            evaluator_args.discard_labels = args.discard_images

            # Prepare yaml file with parameters for identify.
            save_args_to_yaml(identify_yaml_temp, dict(params_set))

            # Run evaluator.
            if not should_skip(hypertune_params, params_set):
                return evaluator_args
            else:
                print("Results for {0} exist, skipping...".format(name))
                return None

        evaluator_args = []
        # First prepare all params sets to run.
        for params_set in tqdm(params_sets_to_run, desc="Preparing param sets to run..."):
            description = hypertune_params.get_description(params_set)
            evaluator_arg = prepare_evaluator_args(hypertune_params, params_set)
            if evaluator_arg is not None:
                evaluator_args.append((description, evaluator_arg))

        # Run one to get predictions.
        if evaluator_args:
            produced_result_dirnames.append(run_one_parameter_set(evaluator_args[0]))

        if args.workers > 1:
            # Rest can run in parallel.
            with Pool(args.workers) as pool:
                produce_results(evaluator_args, pool, produced_result_dirnames)
        else:
            produce_results(evaluator_args, pool=None, produced_result_dirnames=produced_result_dirnames)

        # Run compare.py to get comparison.
        results_to_compare = produced_result_dirnames if not args.report_all else None
        run_comparison(args.output, results_to_compare)

        # Print summary file to console, sorted by F-score.
        report_path = os.path.join(args.output, "Sensible Report.csv")
        with open(report_path, "r") as csv_file:
            csv_results = list(csv.reader(csv_file))
            for row in csv_results[:1] + sorted(csv_results[1:], key=lambda r: float(r[3])):
                print("{0:<30} {1:<30} {2:<30} {3:<30}".format(row[0], row[1], row[2], row[3]))

    finally:
        shutil.rmtree(temp_dir)


def produce_results(evaluator_args, pool, produced_result_dirnames):
    if pool is not None:
        labels_paths = pool.imap(run_one_parameter_set, evaluator_args[1:])
    else:
        labels_paths = map(run_one_parameter_set, evaluator_args[1:])

    for labels_path in tqdm(labels_paths,
                            desc="Parameters evaluation...", miniters=1,
                            total=len(evaluator_args) - 1):
        sys.stderr.flush()
        produced_result_dirnames.append(labels_path)


def tune_components(args):
    fixed_params = OrderedDict([
        ('method', 'components'),
        ('dilation', 2)
    ])
    params_to_test = OrderedDict([
        ('opening', [0, 1, 2]),
        ('threshold', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ])
    hypertune_params = HypertuneParams(fixed_params, params_to_test)
    hypertune_params.abbrevs['method'] = None
    hypertune_params.abbrevs['opening'] = 'open'
    hypertune_params.abbrevs['threshold'] = 'thresh'

    tune(args, hypertune_params)


def tune_watershed(args):
    fixed_params = OrderedDict([
        ('method', 'watershed'),
        ('dilation', 2)
    ])

    params_to_test = OrderedDict([
        ('threshold', [0.3, 0.4, 0.45, 0.5, 0.6]), #, 0.7]),
        ('smooth_mask', [1, 3]), #, 5]),
        ('suppress_peaks', [7, 9, 11]), # , 13]),
        ('smooth_distances', [1, 2, 3]),
        ('use_intensity', [True]),
        ('z_step', [0.2, 0.3, 0.4])
        ])
    hypertune_params = HypertuneParams(fixed_params, params_to_test)
    hypertune_params.abbrevs['method'] = None

    hypertune_params.abbrevs['threshold'] = 'thresh'
    hypertune_params.abbrevs['smooth_mask'] = 'med'
    hypertune_params.abbrevs['suppress_peaks'] = 'peak'
    hypertune_params.abbrevs['smooth_distances'] = 'smo'
    hypertune_params.abbrevs['use_intensity'] = 'intense'
    hypertune_params.abbrevs['z_step'] = 'zs'

    tune(args, hypertune_params)


if __name__ == '__main__':
    args = parse_arguments()

    if args.method == 'components':
        tune_components(args)
    elif args.method == 'watershed':
        tune_watershed(args)
    else:
        raise Exception("unknown method")
