import os
from collections import OrderedDict

import numpy as np
import pandas
import sklearn.model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    make_scorer, roc_auc_score

METRICS_LIST = [
    ("Accuracy", accuracy_score), ("Precision", precision_score), ("Recall", recall_score), ("F", f1_score),
    ("Average precision", average_precision_score), ("AUROC", roc_auc_score)
]

SCORER_LIST = [
    ("Accuracy", make_scorer(accuracy_score)),
    ("Precision", make_scorer(precision_score)), ("Recall", make_scorer(recall_score)), ("F", make_scorer(f1_score)),
    ("Average precision", make_scorer(average_precision_score, needs_proba=True)),
    ("AUROC", make_scorer(roc_auc_score, needs_proba=True))
]


def evaluate_model(model, x_y, metrics_list=METRICS_LIST):
    """
    Evaluate the trained model on the provided datasets (e.g. unseen chunk) and return calculated metrics.
    Args:
        model: trained classificator
        x_y: pair of input and output data
        metrics_list: list of (name, score_function) pairs which specify which metrics are to be calculated.

    Returns:
        dictionary of the scores calculated with metrics extended by the ratio of positive to negative samples
    """
    x, y_gt = x_y
    y_res = model.predict(x)

    positive_class_ratio = np.count_nonzero(y_gt) / len(y_gt)
    res = OrderedDict([("Positive ratio", positive_class_ratio)])
    res.update((name, fun(y_gt, y_res)) for name, fun in metrics_list)

    return res


def cross_validate(regr, x_y, scorers_list=SCORER_LIST, validate_params=None, seed=None, grouping_fn=None):
    """
    Evaluate classification process using cross validation.
    Args:
        regr: configured classificator to be evaluated
        x_y: pair of input and output data
        metrics_list: list of (name, score_function) pairs which specify which metrics are to be calculated.
        validate_params: additional params passed to cross_validate
        seed: seed used for folds 
        grouping_fn (callable): Callable that take as input x and y and return train and test indecies. 

    Returns:
        dictionary of the scores calculated with metrics for each of the folds
    """
    if seed:
        np.random.seed(seed)

    x, y = x_y
    validate_params = dict(validate_params) or {}

    assert not('groups' in validate_params and grouping_fn), 'One grouping method is permitted.'
    if 'groups' not in validate_params and 'group' in validate_params.get('cv', '').__class__.__name__.lower():
        validate_params['groups'] = grouping_fn(x=x)
    
    scorer_dict = dict(scorers_list)
    if 'scoring' not in validate_params:
        validate_params['scoring'] = scorer_dict
    
    scores = sklearn.model_selection.cross_validate(regr, x, y, **validate_params)
    return scores


def get_scores_summary(scores, score_list=None, only_test=True):
    """
    Prepare filtered and averaged summary of the calculated scores.
    Args:
        scores: dictionary of the calculated scores
        score_list: optional list of the scores which should appear in the summary
        only_test: determines if only scores for test sets should appear in the summary

    Returns:
        dictionary of the chosen scores average in case of multiple runs
    """
    res = OrderedDict()
    precise_score_list = None
    if score_list is not None:
        precise_score_list = ["test_" + score for score in score_list]
        if not only_test:
            precise_score_list += ["train_" + score for score in score_list]

    ordered_scores = scores.items() if isinstance(scores, OrderedDict) else sorted(scores.items())
    for k, v in ordered_scores:
        v = np.array(v)  # in case it is a single value
        if precise_score_list is None or k in precise_score_list:
            res[k] = "%0.2f (+/- %0.2f)" % (v.mean(), v.std() * 2)
    return res
