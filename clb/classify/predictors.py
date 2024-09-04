from abc import ABC, abstractmethod

from skimage.filters import threshold_otsu

import clb.classify.train as features_train
import clb.classify.nn.train as dl_train
import clb.classify.nn.predict_cube as dl_predict
from clb.yaml_utils import yaml_file, load_yaml_args


class PredictorBase(ABC):
    """
    This is a base class for predictors capable of the predicting using dictionary with features or object crops.
    It is a common wrapper for both sklearn and keras binary models.

    Attributes:
        model: loaded ready to use model for prediction.
        model_path: path to the model which should be used for predictions
        prob_threshold: threshold used to convert probability to binary decision,
                            default=0.5 the more the probable positive objects are.
    """

    def __init__(self):
        self.model = None
        self.model_path = None
        self.prob_threshold = 0.5

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

    def predict(self, features, crop_data):
        """
        Threshold predict_proba to binary decision.
        """
        return self.predict_proba(features, crop_data) > self.prob_threshold

    @abstractmethod
    def predict_proba(self, features, crop_data):
        """
        Predict the probability of each object as being positive.
        Args:
            features: DataFrame in format with columns used in model training
            crop_data: dictionary of crops around each object

        Returns:
            2d numpy array with two columns: negative_prob, positive_prob.
        """
        pass

    def predict_discrete(self, features, crop_data, discrete=None):
        """
        Use provided model to calculate prediction on given data.
        Args:
            features: DataFrame in format with columns used in model training
            crop_data: crops around each object
            discrete: if None probabilities will be return,
                      the options are:
                        - 'binary': use predict method to get binary classification
                        - '4bins': use otsu thresholding to split probabilities into
                            four classes: sure_no, seems_no, seems_yes, sure_yes

        Returns:
            list of probabilites of the object to be positive according to provided model
        """
        if discrete is None:
            prediction_prob = self.predict_proba(features, crop_data)
            return [p_list[-1] for p_list in prediction_prob]
        elif discrete == 'binary':
            return self.predict(features, crop_data)
        elif discrete == '4bins':
            yes_prob = self.predict_proba(features, crop_data)[:, 1]
            on_off = threshold_otsu(yes_prob)
            sure_off = threshold_otsu(yes_prob[yes_prob < on_off])
            sure_on = threshold_otsu(yes_prob[yes_prob > on_off])

            res = yes_prob.copy()
            res[yes_prob < sure_off] = 0.1
            res[(sure_off <= yes_prob) & (yes_prob < on_off)] = 0.25
            res[(on_off <= yes_prob) & (yes_prob < sure_on)] = 0.75
            res[sure_on <= yes_prob] = 0.9
            return res
        else:
            raise Exception("Not supported discrete method " + discrete)


class FeaturePredictor(PredictorBase):
    """
    Implementation of PredictorBase using sklearn backend and using only features for prediction.
    """

    @classmethod
    def load(cls, path):
        predictor = FeaturePredictor()
        predictor.model = features_train.load_model(path)
        predictor.model_path = path
        return predictor

    def predict_proba(self, features, crop_data):
        return self.model.predict_proba(features)

    def predict(self, features, crop_data):
        return self.model.predict(features)


class DLPredictor(PredictorBase):
    """
    Implementation of PredictorBase using keras backend and using raw crops imagery for prediction.
    """

    @classmethod
    def load(cls, path):
        predictor = DLPredictor()
        predictor.model = dl_train.load_model_with_cache(path)
        predictor.model_path = path
        return predictor

    def predict_proba(self, features, crop_data):
        return dl_predict.predict_cube(self.model, self.model_path, crop_data)


def load_predictor(path):
    """
    Create proper predictor using provided path.
    Args:
        path: filepath to model (h5 or pkl)

    Returns:
        FeaturePredictor or DLPredictor
    """
    path_yaml = yaml_file(path)
    method = load_yaml_args(path_yaml).get('method', 'ExtFeat')
    if method == 'ExtFeat':
        return FeaturePredictor.load(path)
    elif method == 'DL':
        return DLPredictor.load(path)
    else:
        raise Exception("Not supported predictor method " + method)
