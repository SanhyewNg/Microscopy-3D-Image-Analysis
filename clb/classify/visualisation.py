import imageio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation
import daiquiri
from typing import Mapping

from clb.dataprep.utils import rescale_to_float
from clb.image_processing import remove_gaps_in_slices
from clb.utils import replace_values


logger = daiquiri.getLogger(__name__)


class ClassificationVolume:
    @classmethod
    def create(cls, labels_volume: np.ndarray, cell_classes: Mapping[int, float], rescale: bool=True, binary: bool=False) -> np.ndarray:
        """
        Paint the labels according to their class on an empty volume of the same size.
        It ignores labels which have no class assigned.
        Args:
            labels_volume: S x Y x X volume
            cell_classes: index to class or probability dictionary
            rescale: rescale classes values into entire range
            binary: result volume 

        Returns:
            volume where each pixels is colour according its cell class
            if binary set to True result volume positive pixels represent classified cells
        """
        assert len(cell_classes) == 0 or min(cell_classes.values()) >= 0 and max(
            cell_classes.values()) <= 1, "Only binary or probability classes supported."

        values = np.array(list(cell_classes.values()))

        if rescale and any(values) and values.min() != values.max():
            min_val, max_val = values.min(), values.max()
        else:
            min_val, max_val = 0, 1

        rescaled_values = np.interp(values, (min_val, max_val), (2, 255))
        mapping = dict(zip(cell_classes.keys(), rescaled_values))

        res = replace_values(labels_volume, mapping, return_copy=True, zero_unmapped=True)

        if binary:
            res = cls._to_binary(res)

        return res.astype(np.uint8)

    @classmethod
    def from_file(cls, path: str, binary: bool=False) -> np.ndarray:
        """Load classification volume from given path.
        
        Args:
            path: path to classification volume (image)
            binary: translate classification to binary mapping
        Returns:
            volume where each pixel is color according to its cell class
            if binary set to True result volume positive pixels represent classified cells
        """
        classify_volume = imageio.volread(path)
        cls._validate(classify_volume)

        if binary:
            return cls._to_binary(classify_volume)

        return classify_volume

    @classmethod
    def from_array(cls, array: np.ndarray, binary: bool=False) -> np.ndarray:
        """Load classification volume from array.
        
        Args:
            array: input array
            binary: translate classification to binary mapping
        Returns:
            volume where each pixel is color according to its cell class
            if binary set to True result volume positive pixels represent classified cells
        """
        cls._validate(array)

        if binary:
            return cls._to_binary(array)

        return array

    @classmethod
    def _validate(cls, classify_volume: np.ndarray) -> None:
        assert classify_volume.ndim in (2, 3), f"Dimension of classify volume should be 2 or 3. Got: {classify_volume.ndim}"
        assert (np.amax(classify_volume) <= 255) and (np.amin(classify_volume) >= 0), f"Values of ClassificationVolume should fit: 0 <= x <= 255." 

    @classmethod
    def _to_binary(cls, classify_volume: np.ndarray) -> np.ndarray:
        binary_mask =  classify_volume > 128
        return binary_mask.astype(np.uint8)

def show_class_prediction_for_volume(volume_set, class_name, channels, volume_gt_pred):
    """
    Visualize the classification results as a figure and store it to file.
    Args:
        volume_set: ClassVolumeSet which to visualize
        class_name: classification name
        channels: list of channels for which features were calculated. Visualisation present only those channels.
        volume_gt_pred: DataFrame with volume_name as index and columns 'class' (gt) and 'pred' (prediction) with
                        the class.
    """
    input_volume = imageio.volread(volume_set.input)
    labels_volume = imageio.volread(volume_set.label)
    labels_volume = remove_gaps_in_slices(labels_volume)

    # Choose middle slice only - ideally it should be the frame that we have annotations for.
    middle_slice = (len(input_volume)) // 2

    input = input_volume[middle_slice]
    labels = labels_volume[middle_slice]

    # Show only used channels + DAPI.
    for c in range(input.shape[-1]):
        if c not in channels:
            input[..., c] = 0

    # Shrink to better see separations.
    #labels = labels * (skimage.segmentation.find_boundaries(labels) == 0)

    gt_for_this_crop = dict(volume_gt_pred['class'].items())
    gt_classes = ClassificationVolume.create(labels, gt_for_this_crop)

    pred_for_this_crop = dict(volume_gt_pred['pred'].items())
    pred_classes = ClassificationVolume.create(labels, pred_for_this_crop)

    assert len(gt_for_this_crop) == len(pred_for_this_crop)

    # Arrange and save using matplotlib to vol.classes.
    res_figure = show_results(input, labels, gt_classes, pred_classes)
    res_figure.savefig(volume_set.classes[:-3] + "_" + class_name + ".tif")


def show_results(input, labels, gt, classes):
    """
    Present the one slice classification results as a 2x2 Matplotlib Figure.
    Args:
        input: single or multi-channel input image
        labels: single channel instance segmentation label image
        gt: single channel image representing ground truth classes
        classes: single channel image representing predicted classes
    Returns:
        2x2 figure with all the data
    """
    input_plot = rescale_to_float(input)
    random_cmap = colors.ListedColormap([(0, 0, 0)] + list(np.random.rand(256, 3)))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(input_plot)
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(2, 2, 2)
    plt.imshow(labels, cmap=random_cmap)
    plt.axis('off')
    plt.title('Instance segmentation')

    cmap = plt.get_cmap('RdYlGn')
    cmap.set_under('black')

    plt.subplot(2, 2, 3)
    plt.imshow(gt, cmap=cmap, vmin=1, vmax=255)
    plt.title('Ground-truth classes')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(classes, cmap=cmap, vmin=1, vmax=255)
    plt.title('Predicted classes')
    plt.axis('off')

    fig = plt.gcf()
    # plt.show()
    return fig
