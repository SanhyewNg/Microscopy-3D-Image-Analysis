import pytest
import unittest
from collections import OrderedDict

import numpy as np

from clb.segment.hypertune_segment_cells import HypertuneParams


@pytest.mark.hypertuning
class TestHypertuneParams(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def test_hypertune_empty(self):
        params = HypertuneParams({}, {})
        self.assertEqual([{}], list(params.get_all_complete_params_sets()))

    def test_hypertune_of_only_one_param(self):
        params = HypertuneParams({}, {'stack': [1, 2, 3]})
        params_sets = list(params.get_all_complete_params_sets())
        self.assertEqual([{'stack': 1}, {'stack': 2}, {'stack': 3}], params_sets)
        self.assertEqual('stack=1', params.get_description(params_sets[0]))
        self.assertEqual('stack_1', params.get_file_prefix(params_sets[0]))

        params_with_basic = HypertuneParams({'method': 'eye'}, {'stack': [1, 2]})
        params_sets_with_basic = list(params_with_basic.get_all_complete_params_sets())
        self.assertEqual([{'method': 'eye', 'stack': 1}, {'method': 'eye', 'stack': 2}], params_sets_with_basic)
        self.assertEqual('method=eye, stack=1', params_with_basic.get_description(params_sets_with_basic[0]))
        self.assertEqual('metho_eye_stack_1', params_with_basic.get_file_prefix(params_sets_with_basic[0]))

    def test_hypertune_of_two_params(self):
        def order_dict(dict):
            sorted_pairs = sorted(dict.items(), key=lambda x: ['stack', 'pen', 'size'].index(x[0]))
            return OrderedDict(sorted_pairs)

        params = HypertuneParams({}, order_dict({'stack': [1, 2, 3], 'pen': ['blue'], 'size': ['small', 'big']}))
        params_sets = list(params.get_all_complete_params_sets())
        expected_dicts = [{'stack': 1, 'pen': 'blue', 'size': 'small'}, {'stack': 1, 'pen': 'blue', 'size': 'big'},
                          {'stack': 2, 'pen': 'blue', 'size': 'small'}, {'stack': 2, 'pen': 'blue', 'size': 'big'},
                          {'stack': 3, 'pen': 'blue', 'size': 'small'}, {'stack': 3, 'pen': 'blue', 'size': 'big'}]

        self.assertEqual(list(map(order_dict, expected_dicts)), params_sets)
        self.assertEqual('stack=1, pen=blue, size=small', params.get_description(params_sets[0]))
        self.assertEqual('stack_1_pen_blue_size_small', params.get_file_prefix(params_sets[0]))

        params.abbrevs['stack'] = 'st'
        self.assertEqual('stack=1, pen=blue, size=small', params.get_description(params_sets[0]))
        self.assertEqual('st_1_pen_blue_size_small', params.get_file_prefix(params_sets[0]))

        params.abbrevs['size'] = None
        self.assertEqual('stack=1, pen=blue, size=small', params.get_description(params_sets[0]))
        self.assertEqual('st_1_pen_blue', params.get_file_prefix(params_sets[0]))

    def tearDown(self):
        pass
