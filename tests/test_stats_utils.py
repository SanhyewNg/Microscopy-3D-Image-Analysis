import pytest
import unittest

import clb.stats.utils as stutils
import numpy as np


@pytest.mark.statistics
class TestMapIdsToClassGroups(unittest.TestCase):
    def test_output(self):
        all_ids = {1, 2, 3, 4, 5}
        classes_to_ids = {'epith': {1, 2}, 'ki67': {2}, 'cd31': {3, 4}}

        expected_ids_to_groups = {
            1: 'epith',
            2: 'epith, ki67',
            3: 'cd31',
            4: 'cd31',
            5: ''
        }
        ids_to_groups = stutils.map_ids_to_class_groups(all_ids,
                                                        classes_to_ids)

        self.assertEqual(ids_to_groups, expected_ids_to_groups)


@pytest.mark.statistics
class TestGetClassIds(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2]
        ])

    def test_empty_classes(self):
        classes = np.zeros((4, 4))

        class_ids = stutils.get_class_ids(self.labels, classes)
        expected_class_ids = set()

        self.assertEqual(class_ids, expected_class_ids)

    def test_non_empty_classes(self):
        classes = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        class_ids = stutils.get_class_ids(self.labels, classes)
        expected_class_ids = {1}

        self.assertEqual(class_ids, expected_class_ids)


@pytest.mark.statistics
class TestGetAllIds(unittest.TestCase):
    def test_empty_volume(self):
        labels = np.zeros((4, 4))

        all_ids = stutils.get_all_ids(labels)
        expected_all_ids = set()

        self.assertEqual(all_ids, expected_all_ids)

    def test_non_empty_volume(self):
        labels = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 3]
        ])

        all_ids = stutils.get_all_ids(labels)
        expected_all_ids = {1, 2, 3}

        self.assertEqual(all_ids, expected_all_ids)


@pytest.mark.statistics
class TestMapClassesToIds(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 0, 0, 3],
            [0, 0, 0, 0]
        ])
        self.epith = np.array([
            [  2,   2, 218, 218],
            [  2,   2, 218, 218],
            [127,   2,   2, 255],
            [  2,   2,   2,   2]
        ])

    def test_one_class_volume(self):
        classes_to_ids = stutils.map_classes_to_ids(self.labels,
                                                    epith=self.epith)
        expected_classes_to_ids = {
            'epith': {1, 3},
        }

        self.assertEqual(classes_to_ids, expected_classes_to_ids)

    def test_two_class_volumes(self):
        ki67 = np.array([
            [  2,   2,  58,  58],
            [  2,   2,  58,  58],
            [198,   2,   2, 255],
            [  2,   2,   2,   2]
        ])

        classes_to_ids = stutils.map_classes_to_ids(self.labels,
                                                    epith=self.epith,
                                                    ki67=ki67)
        expected_classes_to_ids = {
            'epith': {1, 3},
            'ki67': {2, 3},
        }

        self.assertEqual(classes_to_ids, expected_classes_to_ids)


@pytest.mark.statistics
class TestMapClassesCombsToIds(unittest.TestCase):
    def setUp(self):
        self.all_labels = {1, 2, 3}

    def test_one_class(self):
        classes_to_ids = {
            'epith': {1, 2}
        }

        classes_combs_to_ids = stutils.map_class_groups_to_ids(self.all_labels,
                                                               classes_to_ids)
        expected_classes_combs_to_ids = {
            'all_cells': {1, 2, 3},
            'no_class': {3},
            'epith': {1, 2}
        }

        self.assertEqual(classes_combs_to_ids, expected_classes_combs_to_ids)

    def test_two_classes(self):
        classes_to_ids = {
            'epith': {1, 2},
            'ki67': {2, 3}
        }

        classes_combs_to_ids = stutils.map_class_groups_to_ids(self.all_labels,
                                                               classes_to_ids)
        expected_classes_combs_to_ids = {
            'all_cells': {1, 2, 3},
            'no_class': set(),
            'epith': {1, 2},
            'ki67': {2, 3},
            'epith, ki67': {2}
        }

        self.assertEqual(classes_combs_to_ids, expected_classes_combs_to_ids)
