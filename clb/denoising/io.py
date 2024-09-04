"""Module defines tools for reading and writing data for denoising."""
import glob
import itertools as it
import os
import random

import cv2
import skimage


def read_png(path):
    """Read .png file and return it as tensor with given `dtype`.

    Args:
        path (str): Path to file.

    Returns:
        np.ndarray: Read image, float32.
    """
    image = skimage.img_as_float32(cv2.imread(path, cv2.IMREAD_UNCHANGED))

    return image


def list_files(file_pattern, shuffle, seed=3):
    """List all paths that match given `file_pattern`, optionally shuffle them.

    Args:
        file_pattern (str): Pattern for files.
        shuffle (bool): Should paths be shuffled.
        seed (int): Seed for shuffle operation.

    Returns:
        tuple: List with paths and number of them.
    """
    files_paths = glob.glob(file_pattern, recursive=True)

    if shuffle:
        random.seed(seed)
        random.shuffle(files_paths)

    return files_paths


def read_paths_pairs(fov_dir):
    """Return paths of possible input/target images from `fov_dir`.

    Function checks if there is 'groups.txt' file in `fov_dir`. If it doesn't
    exits, then all possible paths pairs are returned. If file exists function
    assumes that it consists of rows <path_1> <path_2>, separated by newlines.

    Args:
        fov_dir (str): Directory to read paths pairs from.

    Returns:
        list: Paths pairs.
    """
    groups_file_path = os.path.join(fov_dir, 'groups.txt')

    if os.path.isfile(groups_file_path):
        paths_pairs = read_given_paths_pairs(fov_dir, groups_file_path)
    else:
        paths_pairs = read_all_paths_pairs(fov_dir)

    return paths_pairs


def generate_pairs(groups_file):
    """Yield pairs of input/target described by `groups_file`.

    Args:
        groups_file (str): Path to file with pairs. Should consist of rows
                           <path_1> <path_2> separated by newlines.

    Yields:
        tuple: First filename, second filename.
    """
    with open(groups_file) as pairs_file:
        for line in pairs_file:
            first, second = line.split()

            yield first, second


def read_given_paths_pairs(fov_dir, groups_file):
    """Read paths pairs from `fov_dir` basing on `groups_file`.

    Args:
        fov_dir (str): Path to directory with files described by `groups_file`.
        groups_file (str): Path to file describing possible pairs.

    Returns:
        list: Pairs of paths.
    """
    filename_pairs = generate_pairs(groups_file)
    paths_pairs = [(os.path.join(fov_dir, first_filename),
                    os.path.join(fov_dir, second_filename))
                   for first_filename, second_filename in filename_pairs]

    return paths_pairs


def read_all_paths_pairs(fov_dir):
    """Return all possible pairs of paths from `fov_dir`.

    Args:
        fov_dir (str): Path to directory with all files.

    Returns:
        list: List of all possible pairs.
    """
    paths = sorted(glob.glob(os.path.join(fov_dir, '*')))
    paths_pairs = list(it.combinations(paths, 2))

    return paths_pairs


def list_fovs(fovs_pattern, shuffle=False, seed=3):
    """Return (possibly shuffled) paths to input/targets.

    Args:
        fovs_pattern (str): Pattern matched by desired fovs.
        shuffle (bool): Should paths be shuffled.
        seed (int): Seed used when shuffling.

    Returns:
        list: Pairs, path to input, path to target.
    """
    fovs_paths = glob.glob(fovs_pattern, recursive=True)
    paths_pairs = list(it.chain.from_iterable(
        read_paths_pairs(fov_path) for fov_path in fovs_paths
    ))

    if shuffle:
        random.seed(seed)
        random.shuffle(paths_pairs)

    return paths_pairs
