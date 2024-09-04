import argparse
import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

import daiquiri
import matplotlib
from orderedattrdict import AttrDict

from clb.classify.classify import get_parser
from clb.classify.classify import main as run_classify
from clb.classify.train import main as run_classify_train
from clb.classify.train import parse_arguments as parse_train
from clb.evaluate.evaluator_classify import main as run_evaluator
from clb.evaluate.evaluator_classify import \
    parse_arguments as get_evaluator_classify_parser
from clb.yaml_utils import load_yaml_args, yaml_file

matplotlib.use('Agg')

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def get_relative_model_path(model):
    return os.path.join("models", "classification", model)


def get_instance_model_name(train_yaml_path):
    train_args = load_yaml_args(train_yaml_path)
    model_filename = os.path.basename(train_args.instance_model)
    return os.path.splitext(model_filename)[0]


def retraining_possible(model_path):
    train_args = load_yaml_args(yaml_file(model_path))
    return 'ExtFeat' == train_args.get('method', 'ExtFeat')


def run_retrain_full(class_name, output, model, data, datasets, cross_validate='StratifiedKFold'):
    logger.info("Retraining from scratch using: {}".format(model))
    results_dir = os.path.join(output, "results", "train_test_results")
    os.makedirs(results_dir, exist_ok=True)

    model_name = os.path.splitext(model)[0]
    model_path = get_relative_model_path(model)
    instance_segment_model = get_instance_model_name(yaml_file(model_path))
    labels_path = str(Path(output) / "segmented" / instance_segment_model)

    params = [
        "--class_name", class_name,
        "--saved_args", yaml_file(model_path),
        "--data", data,
        "--datasets", datasets,
        "--labels", labels_path,
        "--eval_path", os.path.join(results_dir, class_name, model_name),
        "--cross_validate", cross_validate
    ]

    args = parse_train(params)
    run_classify_train(args)


def run_evaluate(class_name, model, training_data_root, output, *,
                 cross_validate='StratifiedKFold', regenerate=False, data=None, datasets=None):
    os.makedirs(output, exist_ok=True)

    training_data = None
    if training_data_root is not None:
        training_data = [f for f in glob.iglob(os.path.join(training_data_root, '*.tsv'))
                         if os.path.splitext(f)[0].endswith(class_name)][0]

    model_name = os.path.splitext(model)[0]
    evaluation_output_path = os.path.join(output, model_name)
    params = [
        "--class_name", class_name,
        "--saved_args", yaml_file(get_relative_model_path(model)),
        "--eval_path", evaluation_output_path,
        "--cross_validate", cross_validate
    ]

    if training_data is not None:
        params += ["--training_data", training_data]

    if regenerate:
        params += [
            "--data", data,
            "--datasets", datasets,
            "--labels", os.path.join(evaluation_output_path, "labels")
        ]

    args = parse_train(params)
    run_classify_train(args)
    return evaluation_output_path


def run_retrain_compare(class_name, models_to_evaluate, training_data, output, *,
                        cross_validate='StratifiedKFold', regenerate=False, data=None, datasets=None):
    logger.info("Running retrain compare on {} models".format(len(models_to_evaluate)))
    compare_params = []
    results_dir = os.path.join(output, "results", "cross_predict_results", class_name)
    for model in models_to_evaluate:
        logger.info("Retraining from features using: {}".format(model))
        evaluation_path = run_evaluate(class_name, model, training_data, results_dir,
                                       regenerate=regenerate, data=data, datasets=datasets,
                                       cross_validate=cross_validate)
        rel_path_to_output = os.path.relpath(os.path.join(evaluation_path, "Output"), results_dir)
        compare_params.append(rel_path_to_output)

    if len(compare_params) > 1:
        compare_call = 'python vendor/ep/compare.py ' + \
                       results_dir + ' ' + \
                       " ".join(compare_params)
        logger.info("Calling retrain compare: {}".format(compare_call))
        subprocess.run(compare_call, stderr=subprocess.STDOUT, shell=True, check=True,
                       bufsize=0)
    else:
        # If single copy that one so that it has the same result as a comparison.
        single_results_path = os.path.join(results_dir, compare_params[0])
        single_summary = glob.glob(os.path.join(single_results_path, "*.summary.txt"))[0]
        shutil.copy(single_summary, os.path.join(results_dir, "Summary.txt"))


def run_evaluate_compare(output, data, datasets, eval_models):
    logger.info("Running evaluate compare on {} models".format(sum(list(map(len, eval_models.values())))))
    output = os.path.join(output, "results", "classify_evaluate_results")
    for class_name, models in eval_models.items():
        classifier_output_to_compare = []
        for model in models:
            call_evaluator(model=model,
                           output=output,
                           class_name=class_name,
                           data=data,
                           datasets=datasets,
                           labels=None,
                           overlap_treshold=0.5)
            classifier_output_to_compare.append(os.path.join(Path(model).stem, "Output"))
            compare_output = os.path.join(output, class_name)
        if len(classifier_output_to_compare) > 1:
            compare_call = " ".join(['python vendor/ep/compare.py', compare_output, *classifier_output_to_compare])
            logger.info("Calling EP compare: {}".format(compare_call))
            subprocess.run(compare_call, stderr=subprocess.STDOUT, shell=True, check=True, bufsize=0)
        else:
            # If single copy that one so that it has the same result as a comparison.
            single_results_path = os.path.join(output, classifier_output_to_compare[0])
            single_summary = glob.glob(os.path.join(single_results_path, "*.summary.txt"))[0]
            dest_summary_path = os.path.join(output, class_name, "Summary.txt")
            logger.info("Only one model to compare - copy model summary to output: {}".format(dest_summary_path))
            shutil.copy(single_summary, dest_summary_path)


def call_evaluator(model, output, class_name, data, datasets, labels, overlap_treshold):
    """Create an evaluator_classify call based on parameters"""

    model_name = Path(model).stem
    evaluation_output = str(Path(output) / class_name)
    model_path = get_relative_model_path(model)
    model_conf_path = str(Path(model_path).with_suffix(".yaml"))
    instance_model = get_instance_model_name(model_conf_path)
    instance_model_name = Path(instance_model).stem
    labels = labels if labels else str(
        Path(output).parents[1] / "segmented" / instance_model_name)  # if not given, labels set to `root` output folder

    params = [
        "--model", model_path,
        "--name", model_name,
        "--output", evaluation_output,
        "--class_name", class_name,
        "--data", str(Path(data).resolve()),
        "--datasets", datasets,
        "--labels", labels,
        "--overlap_treshold", "0.5",
    ]

    parser = get_evaluator_classify_parser(params)
    kwargs = dict(parser._get_kwargs())

    try:
        run_evaluator(**kwargs)
    except Exception:
        raise RuntimeError('clb.evaluator_classify for model {} failed.'.format(model))


def run_solution(class_name, output, model, data, suffix=''):
    """
    Run entire solution using provided model on a sample image.
    Args:
        output: path where to store results
        model: model to use
        data: root folder for input imagery
        suffix: suffix to add to output files
    """
    logger.info("Running solution using: {}".format(model))

    results_dir = os.path.join(output, "results", "classify_test")
    os.makedirs(results_dir, exist_ok=True)

    all_tiff = sorted(glob.iglob(os.path.join(data, '**/*.tif'), recursive=True))
    non_annotation_tiff = [f for f in all_tiff if not f.endswith("_shapes.tif")]

    model_name = os.path.splitext(model)[0]
    model_path = get_relative_model_path(model)
    instance_segment_model = get_instance_model_name(yaml_file(model_path))
    test_file_prefix = "classes_test_{0}_{1}_{2}".format(suffix, class_name, model_name)
    test_file_input = "classes_test_{0}_input.tif".format(suffix)
    test_file_segment = "classes_test_{0}_segment_{1}.tif".format(suffix, instance_segment_model)
    test_file_output = test_file_prefix + "_output.tif"

    shutil.copy(non_annotation_tiff[0], os.path.join(results_dir, test_file_input))

    parser = get_parser()
    args = parser.parse_args([
        '--input', os.path.join(results_dir, test_file_input),
        '--outputs', os.path.join(results_dir, test_file_output),
        '--model', model_path,
        '--labels', os.path.join(results_dir, test_file_segment)
    ])
    logger.info("args {}".format(args))
    run_classify(args)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare multiple models.',
                                     add_help=False)

    required = parser.add_argument_group('required arguments')
    required.add_argument('--output', help='output directory for results',
                          required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--data', help='path to directory with train, val and test dirs')
    optional.add_argument('--datasets', help='datasets to be used: e.g. "train+val", "test"')
    optional.add_argument('--markers', help='markers to test: "pdl1_cd8" or "epith_ki67"')
    optional.add_argument('--regenerate_all', action='store_true',
                          help='recalculate everything from scratch (very time consuming)')
    optional.add_argument('--regression_all', action='store_true',
                          help='run classify and retraining for each model (may be time consuming)')
    optional.add_argument('--training_data_root', help='path to folder with generated training data')
    optional.add_argument('--cross_validate', choices=['StratifiedKFold', 'KFold', 'GroupKFold', 'GroupShuffleSplit'],
                          help='Specify type of cross-validation. When `Group` type selected, samples will be split by image.',
                          default='StratifiedKFold')
    return parser.parse_args()


def filter_loaded_models(loaded_models, args_class_names, all_models=False, retraining=False):
    sorted_models_to_eval = AttrDict()
    for class_name, models in sorted(loaded_models.items()):
        if args_class_names is not None and class_name not in args_class_names.split("_"):
            continue
        else:
            proper_models = [m for m in models if retraining_possible(get_relative_model_path(m))] if retraining else models
            sorted_models_to_eval[class_name] = proper_models if all_models else [proper_models[-1]]
    return sorted_models_to_eval


if __name__ == '__main__':
    args = parse_arguments()
    models_to_evaluate = load_yaml_args('models/classification/classificators_to_evaluate.yaml')
    selected_models_trainable = filter_loaded_models(models_to_evaluate, args.markers,
                                                     all_models=args.regression_all, retraining=True)
    models_trainable = filter_loaded_models(models_to_evaluate, args.markers,
                                            all_models=True, retraining=True)
    all_models = filter_loaded_models(models_to_evaluate, args.markers,
                                      all_models=True)

    if args.data and args.datasets:
        run_evaluate_compare(output=args.output,
                             data=args.data,
                             datasets=args.datasets,
                             eval_models=all_models)

    if args.data:
        for class_name, models in selected_models_trainable.items():
            for model in models:
                # This is no longer needed as run_evaluate_compare does it better.
                # run_solution(class_name, args.output, model, args.data, args.markers or '')
                if args.datasets:
                    run_retrain_full(class_name, args.output, model, args.data, args.datasets,
                                     cross_validate='StratifiedKFold')  # GroupKFold not feasable - test set is small and group could fail.

    if args.training_data_root or args.regenerate_all:
        for class_name, models in models_trainable.items():
            run_retrain_compare(class_name, models, args.training_data_root, args.output,
                                regenerate=args.regenerate_all, data=args.data, datasets=args.datasets,
                                cross_validate=args.cross_validate)
