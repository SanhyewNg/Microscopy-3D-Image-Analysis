import os
import sys

import imageio
import numpy as np
from skimage import exposure, img_as_int, img_as_ubyte, img_as_uint

import IPython.display
from IPython.display import Markdown, display, HTML
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, NoNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes_classic as marching_cubes

import importlib

# hack to get to ep from here
parent_dir = os.path.split(os.getcwd())[0]
nb_dir = os.path.join(parent_dir, "clb")
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

random_colormap = ListedColormap([(0,0,0)] + list(np.random.rand(20000, 3)))


def pick_path(paths):
    correct_paths = [p for p in paths if os.path.exists(p)]
    if len(correct_paths) > 1:
        raise Exception("More than one existing path to choose.")
    elif len(correct_paths) == 0:
        raise Exception("No path exists.")
    return correct_paths[0]


def plot_3d(*images, lims=None):
    """
    Requires %matplotlib notebook to be interactive.
    """

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    lims = lims or images[0].shape

    for i, image in enumerate(images):
        verts, faces = marching_cubes(image)

        mesh = Poly3DCollection(verts[faces], alpha=0.75)
        face_color = np.random.rand(3)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.set_xlim(0, lims[0])
    ax.set_ylim(0, lims[1])
    ax.set_zlim(0, lims[2])

    plt.show()


def show_3d_slices(rows, *args, **kwargs):
    """
    Presents given 3d arrays in form of grid as matplotlib figure.
    Args:
        rows: number of rows which is equal the number of 3d arrays to show.
        *args: list of numpy 3d arrays to show
        **kwargs:
            - scale: define figsize
            - cmap: value mapping for each or every 3d array
                if 'rand' then no normalization and random int to RGB mapping used
                else it is just passed to imshow

    Returns:
        matplotlib figure to show in notebook
    """
    cols = len(args[0])
    all_slices = [array_slice for array_3d in args for array_slice in array_3d]

    if 'cmap' in kwargs:
        cmap_3d = kwargs['cmap'].split()
        cmaps_per_slice = [cmap for cmap, array_3d in zip(cmap_3d, args) for array_slice in array_3d]
        kwargs['cmap'] = ",".join(cmaps_per_slice)

    return show_all(rows, cols, *all_slices, **kwargs)


def show_all(rows, cols, *args, **kwargs):
    """
    Presents given arrays in form of grid as matplotlib figure.
    Args:
        rows: number of rows to show.
        cols: number of columns to show.
        *args: list of numpy arrays to show
        **kwargs: 
            - scale: define figsize
            - cmap: value mapping for each or every array
                if 'rand' then no normalization and random int to RGB mapping used 
                else it is just passed to imshow

    Returns:
        matplotlib figure to show in notebook
    """
    scale = 30
    if 'scale' in kwargs:
        scale = kwargs['scale']

    subtitles = kwargs.get("titles", [""])

    gray_cmaps = ['gray']
    random_map = random_colormap
    normalization = [True]
    if 'cmap' in kwargs:
        gray_cmap_params = kwargs['cmap'].split(',')
        gray_cmaps = []
        normalization = []
        for m in gray_cmap_params:
            is_rand = m.strip() == 'rand'
            normalization.append(not is_rand)
            if is_rand:
                gray_cmaps.append(random_map)
            else:
                gray_cmaps.append(m)

    shape_ratio = args[0].shape[0] / args[0].shape[1]
    fig, axes = plt.subplots(rows, cols, figsize=(scale * cols, scale * rows * shape_ratio))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    for i in range(rows * cols):
        array = args[i] if len(args) > i else np.zeros(args[0].shape)
        if rows == 1 and cols == 1:
            ax = axes
        else:
            ax = axes[i] if rows == 1 or cols == 1 else axes[i//cols][i % cols]

        sub_ix = i
        subtitle = ""
        if len(subtitles) <= i:
            sub_ix = i % cols

        if len(subtitles) > sub_ix:
            subtitle = subtitles[sub_ix]

        ax.set_aspect("auto")
        ax.set_title(subtitle)
        if len(array.shape) == 3:
            if array.dtype == np.uint32 or array.dtype == np.uint16:
                array = (array / np.iinfo(array.dtype).max * 255).astype(np.uint8)
            ax.imshow(array)
        else:
            map_id = i % len(gray_cmaps)
            if normalization[map_id]:
                ax.imshow(array, cmap=gray_cmaps[map_id])
            else:
                ax.imshow(array, norm=NoNorm(), cmap=gray_cmaps[map_id])


    plt.close(fig)
    return fig


def printmd(*text):
    display(Markdown(" ".join([str(t) for t in text])))


def display_width(size_perc):
    display(HTML("<style>.container { width:{0}% !important; }</style>".format(size_perc)))