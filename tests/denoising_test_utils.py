"""Module defines tools used during testing denoising code."""
import unittest.mock as mock

import attrdict
import tensorflow as tf


def make_mocks(*args, **kwargs):
    """Make attrdict with mocks.

    Args:
        args: Arguments should be names of created mocks (strings).
        kwargs: Name of argument will be used as mock name, value as mock.

    Returns:
        AttrDict: Created mocks (names to values).
    """
    mocks = attrdict.AttrDict({arg: mock.Mock() for arg in args})
    mocks.update(kwargs)

    return mocks


def patch_function(test_case, target, return_value=None, spec_set=None):
    """Patch `target` and add cleanup stopping patch to `test_case`.

    Args:
        test_case (unittest.TestCase): Test case for adding cleanup.
        target (str): Target to patch.
        return_value (Any): Value that will be returned by patches function.
        spec_set: Same as spec_set argument in unittest.mock.patch.

    Returns:
        unittest.Mock: Mock used to patch function.
    """
    if return_value is None:
        return_value = mock.Mock()

    new = mock.Mock(return_value=return_value)
    patch_target = mock.patch(target, new, spec_set=spec_set)
    patch_target.start()

    test_case.addCleanup(patch_target.stop)

    return new


def make_list_from_dataset(sess, dataset):
    """Return list with elements of dataset.

    Args:
        sess (tf.Session): Session for calculations.
        dataset (tf.data.Dataset): Dataset to read elements from

    Returns:
        list: Elements of `dataset`.
    """
    dataset_iter = dataset.make_one_shot_iterator()
    get_next_element = dataset_iter.get_next()
    elements = []
    try:
        while True:
            e = sess.run(get_next_element)
            elements.append(e)
    except tf.errors.OutOfRangeError:
        pass

    return elements


class Tensor:
    """Class made just to compare tensors."""
    def __init__(self, sess, t):
        self.sess = sess
        self.t = t

    def __eq__(self, other):
        return self.sess.run(tf.reduce_all(tf.equal(self.t, other)))
