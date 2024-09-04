import os
import shutil
import tempfile
import pytest
import unittest
import unittest.mock as mock

import tensorflow as tf

import clb.denoising.io as denoiseio


@pytest.mark.denoising
class TestListFiles(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.addCleanup(self.sess.close)

        self.dir = tempfile.mkdtemp()

        self.first_png = os.path.join(self.dir, 'first.png')
        self.second_png = os.path.join(self.dir, 'second.png')

        open(self.first_png, 'w').close()
        open(self.second_png, 'w').close()
        open(os.path.join(self.dir, 'first.jpeg'), 'w').close()

        self.addCleanup(shutil.rmtree, self.dir)

    def test_if_gives_right_files(self):
        files = denoiseio.list_files(os.path.join(self.dir, '*.png'),
                                     shuffle=False)
        files = set(files)
        expected_files = {self.first_png, self.second_png}

        self.assertEqual(files, expected_files)


class TestReadingPairs(unittest.TestCase):
    def setUp(self):
        self.fov_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.fov_dir)

        self.fov_1 = os.path.join(self.fov_dir, 'fov_1')
        os.makedirs(self.fov_1)
        open(os.path.join(self.fov_1, '0'), 'w').close()
        open(os.path.join(self.fov_1, '1'), 'w').close()
        open(os.path.join(self.fov_1, '2'), 'w').close()

        self.fov_2 = os.path.join(self.fov_dir, 'fov_2')
        os.makedirs(self.fov_2)
        open(os.path.join(self.fov_2, '0'), 'w').close()
        open(os.path.join(self.fov_2, '1'), 'w').close()
        open(os.path.join(self.fov_2, '2'), 'w').close()
        open(os.path.join(self.fov_2, '3'), 'w').close()

        with open(os.path.join(self.fov_2, 'groups.txt'), 'w') as f:
            print('0 1', file=f)
            print('0 2', file=f)
            print('1 3', file=f)

    def test_if_read_paths_pairs_calls_read_given_paths_pairs(self):
        with\
                mock.patch(target='clb.denoising.io.read_given_paths_pairs')\
                as read_given_paths_pairs:
            denoiseio.read_paths_pairs(self.fov_2)
            read_given_paths_pairs.assert_called_once_with(
                self.fov_2, os.path.join(self.fov_2, 'groups.txt')
            )

    def test_if_read_paths_pairs_calls_read_all_paths_pairs(self):
        with\
                mock.patch(target='clb.denoising.io.read_all_paths_pairs')\
                as read_all_paths_pairs:
            denoiseio.read_paths_pairs(self.fov_1)
            read_all_paths_pairs.assert_called_once_with(
                self.fov_1
            )

    def test_generate_pairs_output(self):
        pairs = denoiseio.generate_pairs(os.path.join(self.fov_2,
                                                      'groups.txt'))
        expected_pairs = {('0', '1'), ('0', '2'), ('1', '3')}

        self.assertEqual(set(pairs), expected_pairs)

    def test_read_given_paths_pairs_output(self):
        groups_path = os.path.join(self.fov_2, 'groups.txt')
        pairs = denoiseio.read_given_paths_pairs(self.fov_2, groups_path)
        expected_pairs = {
            (os.path.join(self.fov_2, '0'), os.path.join(self.fov_2, '1')),
            (os.path.join(self.fov_2, '0'), os.path.join(self.fov_2, '2')),
            (os.path.join(self.fov_2, '1'), os.path.join(self.fov_2, '3'))
        }

        self.assertEqual(set(pairs), expected_pairs)

    def test_read_all_paths_pairs(self):
        pairs = denoiseio.read_all_paths_pairs(self.fov_1)
        expected_pairs = {
            (os.path.join(self.fov_1, '0'), os.path.join(self.fov_1, '1')),
            (os.path.join(self.fov_1, '0'), os.path.join(self.fov_1, '2')),
            (os.path.join(self.fov_1, '1'), os.path.join(self.fov_1, '2')),
        }

        self.assertEqual(set(pairs), expected_pairs)

    def test_list_fovs_output(self):
        paths_pairs = [
            [
                ('0', '1')
            ],
            [
                ('2', '3'),
                ('4', '5')
            ]
        ]
        read_paths_pairs = mock.Mock(side_effect=paths_pairs)
        with mock.patch(target='clb.denoising.io.read_paths_pairs',
                        new=read_paths_pairs):
            pattern = os.path.join(self.fov_dir, '*')
            paths = denoiseio.list_fovs(pattern, False)

            self.assertEqual(paths, [('0', '1'), ('2', '3'), ('4', '5')])
