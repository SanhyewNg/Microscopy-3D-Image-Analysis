import argparse
import logging
import os
import pickle
import sys
from pprint import pprint, pformat

import daiquiri
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import model_selection

from clb.classify.evaluation import (METRICS_LIST, cross_validate,
                                     evaluate_model, get_scores_summary)
from clb.classify.feature_extractor import get_feature_columns
from clb.classify.extractors import extract_channels, parse_channels_preprocessing
from clb.classify.prepare_train import merge_all_features, process_volume_sets
from clb.classify.utils import (get_all_datasets, get_all_multiple_datasets,
                                save_to_tsv, group_by_image)
from clb.classify.visualisation import show_class_prediction_for_volume
from clb.utils import parse_string
from clb.yaml_utils import merge_cli_to_yaml, save_args_to_yaml
from clb.evaluate.evaluator_utils import EpEvalFormatter, EpEvaluation


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def separate_features_and_class(data, features_type, channels_preprocessing_list):
    """
    Split the classification data into feature and ground truth data.
    Args:
        data: DataFrame with all classification data
        features_type: str id of the features that were calculated
        channels_preprocessing_list: list of strings with channels (potentially with preprocessing suffix)

    Returns:
        a pair of:
        - DataFrame with only feature columns
        - Series with only ground truth classification column
    """
    return data[get_feature_columns(data.columns, features_type, channels_preprocessing_list)], data['class']


def filter_uncertain(data):
    return data[(data['class'] <= 1) & (data['area'] >= 5)].copy()


def prepare_volume_sets(args, extract_features):
    """
    Based on the provided command line arguments discover and process chosen datasets.

    Among others ensures that segmentation exists, match segmentation to annotations and (if requested)
    calculate features from input data.

    Args:
        args: command line arguments
        extract_features: if False then feature extraction will not be done

    Returns:
        list of ClassVolumeSet with classification related data already calculated
    """
    if args.data is not None and args.datasets is not None:
        datasets_list = args.datasets.split("+")
        all_data = get_all_multiple_datasets(args.data, datasets_list, "input", "annotations", args.labels,
                                             args.eval_path, args.class_name)
    else:
        all_data = get_all_datasets(args.annotated_input, args.annotated_gt,
                                    args.labels, args.eval_path, args.class_name)

    channels = parse_channels_preprocessing(args.channels)
    process_volume_sets(all_data, args.instance_model, channels, args.features_type if extract_features else None,
                        fill_gaps=args.manual_instance, remove_blobs=args.manual_instance,
                        voxel_resize=not args.no_voxel_resize)
    return all_data


def prepare_estimator(n_estimators, estimator_params, seed=None):
    return RandomForestClassifier(n_estimators, random_state=seed, **estimator_params)


def train_model(regr, x_y):
    """
    Train provided classificator on the random part of the provided data and return the unseen chunk.
    Args:
        regr: configured classificator to be trained
        x_y: pair of input and output data

    Returns:
        pair of trained classificator and (X,Y) test data.
    """
    x, y = x_y

    # use some random split for getting model
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1)

    regr.fit(X_train, y_train)
    return regr, (X_test, y_test)


def calculate_test_predictions(regr, x_y, validate_params=None, seed=None, grouping_fn=None):
    """
    Calculate predictions for the given data but always using the model trained on the other data.
    In practice it splits the data into folds (similar to cross_validate) and for each fold it trains the model
    on the remaining ones and predict class probability on the unseen fold.

    After calculating predictions the best (F1 score wise) threshold is found
    and used to get class prediction.
    Args:
        regr: configured classificator to be evaluated
        x_y: pair of input and output data
        validate_params: additional params passed to cross_val_predict
        seed: seed used for folds
        grouping_fn (callable): Callable that take as input x and y and return train and test indecies. 

    Returns:
        pair of
            - class (binary) predictions on all data using models that did not see that data
            - probability of belonging to class
    """
    if seed:
        np.random.seed(seed)

    validate_params = validate_params or {}
    
    assert not('groups' in validate_params and grouping_fn), 'One grouping method is permitted.'
    if 'groups' not in validate_params and 'group' in validate_params.get('cv', '').__class__.__name__.lower():
        validate_params['groups'] = grouping_fn(x=x_y[0])

    x, y = x_y
    # class_val_predict accept only permutation groups. Shuffle splits not feasable.
    y_res_prob = cross_val_predict(regr, x, y, method='predict_proba', **validate_params)
    y_res_prob = y_res_prob[::, 1]

    return y_res_prob > 0.5, y_res_prob


def parse_estimator_params(unknown_params):
    """
    Given params that were not parsed, check if they are all estimator related and parse them.
    Args:
        unknown_params: not parsed parameters in format provided by argparse

    Returns:
        None if there are params that are not estimator related, if not
            the dictionary from estimator params name (str) to parsed value (str, float, int)
    """
    est_params_names = [k[6:] for k in unknown_params[::2] if k.startswith('--est_')]
    if len(est_params_names) * 2 != len(unknown_params):
        return None
    else:
        est_parsed_values = [parse_string(param) for param in unknown_params[1::2]]

        return dict(zip(est_params_names, est_parsed_values))


def parse_arguments(provided_args=None):
    parser = argparse.ArgumentParser(description='CLB classificator training.', add_help=True)
    required = parser.add_argument_group('required arguments')

    required.add_argument('--class_name', help='name of the class to predict used to select annotations', required=True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--seed', type=int, help='set seed for all random generators', default=42)
    optional.add_argument('--channels', help='channels with optional preprocessings to use e.g. "1,2" or "1,1-equal"')
    optional.add_argument('--features_type', help='identifier of the feature set to extract')
    optional.add_argument('--regen', dest='regen', action='store_true', help='regenerate data stored in training_data')
    optional.add_argument('--model', help='path (with name) to save the model')
    optional.add_argument('--no_voxel_resize', action='store_true',
                          help='should volume be resized to match usual pixel size.')
    optional.add_argument('--cross_validate', choices=['StratifiedKFold', 'KFold', 'GroupKFold', 'GroupShuffleSplit'], 
                          help='Specify type of cross-validation. When `Group` type selected, samples will be split by image.',
                          default='StratifiedKFold')

    evaluation = parser.add_argument_group('evaluation arguments')
    evaluation.add_argument('--saved_args', help='path to yaml file with parameters')
    evaluation.add_argument('--eval_path', help='path to folder where to store the results of the evaluation')

    training = parser.add_argument_group('training parameters (additional parameters with prefix: est_ can be passed)')
    training.add_argument('--n_estimators', type=int, help='number of estimators in RF')
    training.add_argument('--est_<estimator_param>', type=str,
                          help='example parameter to be passed to estimator constructor')

    retrain_from_data = parser.add_argument_group('retrain from processed data arguments')
    retrain_from_data.add_argument('--training_data',
                                   help='path to previously generated training data or a place to store it')

    retrain_from_scratch = parser.add_argument_group('retrain from scratch arguments')
    retrain_from_scratch.add_argument('--data',
                                      help='path to directory with train, val and test dirs, '
                                           'required if --annotated_input or --annotated_gt not specified')
    retrain_from_scratch.add_argument('--datasets',
                                      help='used and required if --data is specified, describes which datasets should be used:'
                                           'e.g. "train+val", "test", "val+test"')

    retrain_from_scratch.add_argument('--annotated_input', help='path to folder with input imagery')
    retrain_from_scratch.add_argument('--annotated_gt', help='path to folder with class annotations')

    retrain_from_scratch.add_argument('--labels', help='path to folder with cell labels')
    retrain_from_scratch.add_argument('--instance_model', help='path to h5 model for instance segmentation')

    retrain_from_scratch.add_argument('--manual_instance', dest='manual_instance', action='store_true',
                                      help="are labels actually from manual annotations")

    parser.set_defaults(no_voxel_resize=False)
    parser.set_defaults(manual_instance=False)
    parser.set_defaults(regen=False)

    parsed_args, unknown = parser.parse_known_args(provided_args)

    # If params saved load from yaml.
    if parsed_args.saved_args is not None:
        parsed_args = merge_with_saved(parsed_args, parsed_args.saved_args)
        parsed_args.no_voxel_resize = getattr(parsed_args, "no_voxel_resize", True)

    if parsed_args.training_data is None and \
            (((parsed_args.annotated_input is None or parsed_args.annotated_gt is None)
              and (parsed_args.data is None or parsed_args.datasets is None))
             or parsed_args.channels is None or (parsed_args.labels is None and parsed_args.instance_model is None)
             or parsed_args.features_type is None):
        print("You need to either provide training_data or tools to generate it from scratch: ", file=sys.stderr)
        print("annotated_input, annotated_gt, channels, labels or instance_model, features_type OR ", file=sys.stderr)
        print("data, datasets, channels, labels or instance_model, features_type", file=sys.stderr)
        exit(0)

    est_params = parse_estimator_params(unknown)
    if est_params is None:
        # there are unknown parameters so show standard error
        parser.parse_args(provided_args)
    else:
        parsed_args.estimator = est_params or getattr(parsed_args, 'estimator', {})

    parsed_args.method = "ExtFeat"
    return parsed_args


def merge_with_saved(args, path_to_saved):
    params_in_yaml_to_override = ['seed',
                                  'model',
                                  'eval_path',
                                  'regen',
                                  'data', 'datasets',
                                  'annotated_input', 'annotated_gt', 'manual_instance',
                                  'labels', 'training_data',
                                  'cross_validate']

    merged_args = merge_cli_to_yaml(path_to_saved, vars(args), params_to_merge=params_in_yaml_to_override)
    if args.class_name != merged_args.class_name:
        raise Exception("Class name mismatch with saved: {0} != {1}".format(args.class_name, merged_args.class_name))

    return merged_args


def main(args):
    np.random.seed(args.seed)

    channels_preprocess_list = parse_channels_preprocessing(args.channels)

    # If we want to store models lets save params.
    if args.model:
        model_path_prefix = os.path.splitext(args.model)[0] + "_class_" + args.class_name
        save_args_to_yaml(model_path_prefix, args)

    # Interpret training_data.
    training_datas = None
    if args.training_data is not None:
        training_datas = args.training_data.split("+")
    single_training_data_not_existing = training_datas is not None and \
                                        len(training_datas) == 1 and not os.path.isfile(training_datas[0])

    # Load classification data from file or regenerate it from scratch.
    class_volume_sets = None

    if training_datas is None or single_training_data_not_existing or args.regen:
        # Only one or we want to regen.
        class_volume_sets = prepare_volume_sets(args, extract_features=True)
        data_for_training = merge_all_features(class_volume_sets)

        if single_training_data_not_existing:
            os.makedirs(os.path.dirname(args.training_data), exist_ok=True)
            save_to_tsv(data_for_training, args.training_data)
    else:
        if all(map(os.path.isfile, training_datas)):
            if args.data or args.annotated_input:
                class_volume_sets = prepare_volume_sets(args, extract_features=False)

            # Read and concat potentially multiple files.
            datas_for_training = [pandas.read_csv(td, sep='\t', index_col=[0, 1]) for td in training_datas]
            data_for_training = pandas.concat(datas_for_training, join='inner')
        else:
            raise Exception("Multiple training data provided and not all exist.")

    data_for_training = filter_uncertain(data_for_training)

    # shuffle for better splits
    data_for_training = data_for_training.sample(frac=1, random_state=args.seed)
    x_y = separate_features_and_class(data_for_training, args.features_type, channels_preprocess_list)

    classifier = prepare_estimator(args.n_estimators, args.estimator, seed=args.seed)

    # Train and evalute model that we produce.
    model, test_x_y = train_model(classifier, x_y)
    scores_test = evaluate_model(model, test_x_y)
    print("Static random split test results:")
    pprint(get_scores_summary(scores_test))

    if args.model:
        save_model(model, model_path_prefix + ".pkl")

    # Set cross-validation parametes.
    CV = getattr(model_selection, args.cross_validate)
    try:
        cv = CV(n_splits=5, random_state= args.seed)
    except TypeError:
        cv = CV(n_splits=5)
    params_for_cross_fold = {'cv': cv}
 
    # Evaluate solution in general using folds.
    scores_cross = cross_validate(classifier, x_y, validate_params=params_for_cross_fold, seed=args.seed, grouping_fn=group_by_image)
    summary = get_scores_summary(scores_cross, dict(METRICS_LIST).keys(), only_test=False)
    logger.info('{cv_type} cross-validation results: '.format(cv_type=CV.__name__))
    logger.info(pformat(summary))

    if args.eval_path is not None:
        os.makedirs(args.eval_path, exist_ok=True)

        # save training features
        save_to_tsv(x_y[0], os.path.join(args.eval_path, "features_used_{0}.tsv".format(args.class_name)))

        # save summaries and used arguments
        save_args_to_yaml(os.path.join(args.eval_path, "summary_{0}".format(args.class_name)), args)
        summary_path = os.path.join(args.eval_path, "summary_{0}.txt".format(args.class_name))
        with open(summary_path, "w") as f:
            print("{cv_type} cross-validation results:".format(cv_type=CV.__name__), file=f)
            pprint(summary, stream=f)

        # prepare cross-predictions on all data
        y_pred, y_prob = calculate_test_predictions(classifier, x_y, validate_params=params_for_cross_fold,
                                                    seed=args.seed, grouping_fn=group_by_image)
        predicted_classes = data_for_training.copy()
        predicted_classes["pred"] = y_pred.astype(np.uint8)
        predicted_classes["pred_prob"] = y_prob
        prediction_output_path = os.path.join(args.eval_path, "cross_predictions_{0}.tsv".format(args.class_name))
        save_to_tsv(predicted_classes, prediction_output_path)

        print("Best F-score in cross-validation: {0}".format(f1_score(x_y[1], y_pred)))
        with open(summary_path, "a") as f:
            print("Best F-score in cross-validation: {0}".format(f1_score(x_y[1], y_pred)), file=f)

        # prepare EP friendly results
        gt_ep_tsv = "ep_cross_predictions_gt_{0}.tsv".format(args.class_name)
        pred_ep_tsv = "ep_cross_predictions_pred_{0}.tsv".format(args.class_name)
        ep_formatter = EpEvalFormatter(output_gt_filename=gt_ep_tsv, output_predicted_filename=pred_ep_tsv)
        ep_formatter.from_csv(prediction_output_path, args.eval_path)

        # run EP on that data
        output_parent_path, solution_name = os.path.split(args.eval_path)   
        EpEvaluation(gt_filename=gt_ep_tsv, predicted_filename=pred_ep_tsv).run(output_parent_path, solution_name)
      
        # If requested and we have input data we can present class predictions on unseen data.
        if class_volume_sets is not None:
            channels = extract_channels(channels_preprocess_list)
            for vol in class_volume_sets:
                show_class_prediction_for_volume(vol, args.class_name, channels, predicted_classes.ix[vol.crop_name])


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
