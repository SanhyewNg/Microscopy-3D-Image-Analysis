import random
from copy import copy

import numpy as np
import yaml

from clb.utils import crop_image


class CropInfo:
    """
    Contains information about the image crop and allows to easily re-crop on
    different data (e.g. on output labeled data).
    """

    def __init__(self, y, x, z, shape, blob_index=1):
        """
        Args:
            y: top of the crop
            x: left of the crop
            z: layer of the crop
            shape: height, width of the crop
            blob_index: optional information about the label value reserved for blobs (should be 1)
        """
        self.blob_index = blob_index
        self.shape = shape
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.voxel_size = (0.5, 0.5, 0.5)

    @property
    def zyx_start(self):
        return self.z, self.y, self.x

    @property
    def x_end(self):
        return self.x + self.shape[1]

    @property
    def y_end(self):
        return self.y + self.shape[0]

    @property
    def area(self):
        return self.shape[0] * self.shape[1]

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __lt__(self, other):
        return self.z < other.z

    def crop(self, original_data, from_volume=True):
        """
        Re-crop original_data using stored information. Works with 2D and 3D data.
        Args:
            original_data: NxM or NxMxC or HxNxM or HxNxMxC numpy array
            from_volume: original_data is a 3D volume

        Returns:
            specified crop as NxM or NxMxC numpy array
        """
        assert not from_volume or (0 <= self.z < len(original_data)), "Z-slice {0} outside of bounds.".format(self.z)

        original_image = original_data
        # If input is volume select proper layer.
        if original_image.ndim > 2 and from_volume:
            original_image = original_data[self.z]
        else:
            original_image = original_data
        return crop_image(original_image, self.shape, (self.y, self.x), multi_channel=original_image.ndim > 2)

    def overlap(self, other):
        """
        Calculate number of overlapping pixels between this and other CropInfo.
        Args:
            other: CropInfo

        Returns:
            number of overlapping pixels
        """

        if self.z != other.z:
            return 0

        left_x, right_x = max(self.x, other.x), min(self.x_end, other.x_end)
        left_y, right_y = max(self.y, other.y), min(self.y_end, other.y_end)
        return max(0, right_x - left_x) * max(0, right_y - left_y)

    def restrict(self, bounds):
        """
        Restrict info to provided volume bounds.
        Args:
            bounds: Z x Y x X bounds restricting crop_info
        Returns:
            crop restricted to bounds or None if crop is entirely outside of bounds.
        """
        res = copy(self)

        sz, sy, sx = bounds
        if res.z < 0 or res.z >= sz:
            return None

        if res.x >= sx:
            return None
        width = min(sx - res.x, res.shape[1])
        if res.x < 0:
            width += res.x
            res.x = 0
        if width <= 0:
            return None

        if res.y >= sy:
            return None
        height = min(sy - res.y, res.shape[0])
        if res.y < 0:
            height += res.y
            res.y = max(res.y, 0)
        if height <= 0:
            return None

        res.shape = (height, width)
        return res

    @staticmethod
    def empty_volume(crop_infos, dtype=np.uint8):
        """
        Create empty volume of the shape that bound the provided CropInfos
        Args:
            crop_infos: shape that we want to bound

        Returns:
            empty array HxNxM that is a bounding box of the provided CropInfos
        """
        assert CropInfo.is_block(crop_infos)

        zs = [i.z for i in crop_infos]
        shape = crop_infos[0].shape  # we assume that crop_infos form volume
        z_size = max(zs) - min(zs) + 1

        return np.zeros((z_size, shape[0], shape[1]), dtype=dtype)

    @staticmethod
    def overlap_volume(crop_infos, other_crop_infos):
        """
        Calculate number of overlapping pixels between two sets of crop infos.
        Args:
            crop_infos: list of CropInfo
            other_crop_infos: list of CropInfo

        Returns:
            number of overlapping pixels
        """
        other_dict = {i.z: i for i in other_crop_infos}
        overlap = 0
        for info in crop_infos:
            if info.z in other_dict:
                overlap += info.overlap(other_dict[info.z])
        return overlap

    @staticmethod
    def iou_volume(crop_infos, other_crop_infos):
        """
        Calculate intersection over union between two sets of crop infos.
        Args:
            crop_infos: list of CropInfo
            other_crop_infos: list of CropInfo

        Returns:
            intersection over union
        """
        overlap = CropInfo.overlap_volume(crop_infos, other_crop_infos)
        area = sum(i.area for i in crop_infos)
        other_area = sum([i.area for i in other_crop_infos])
        return overlap / (area + other_area - overlap)

    @staticmethod
    def overlap_volume_fraction(new_crop_infos, existing_crop_infos):
        """
        Calculate how much of the new crop infos is already covered by
        the existing crop infos.
        Args:
            new_crop_infos: list of CropInfo
            existing_crop_infos: list of CropInfo

        Returns:
            fraction of the new_crop_infos that overlap with existing_crop_infos
        """
        overlap = CropInfo.overlap_volume(new_crop_infos, existing_crop_infos)
        area = sum([i.area for i in new_crop_infos])
        return overlap / area

    @staticmethod
    def extend_infos(crop_infos, padding_3d):
        """
        Extend list of crop_infos that represents 3d block by specified padding.
        Args:
            crop_infos: list of CropInfo
            padding_3d: (z,y,x) padding

        Returns:
            CropInfos list padded by padding_3d. It can have negative coordinates.
        """
        assert CropInfo.is_block(crop_infos)
        pad_z, pad_y, pad_x = padding_3d

        # We asserted that it is not empty.
        y, x = crop_infos[0].y, crop_infos[0].x
        height, width = crop_infos[0].shape
        zs = [i.z for i in crop_infos]
        z_min, z_max = zs[0], zs[-1]

        # Calculate new infos block position and size.
        new_zs = range(z_min - pad_z, z_max + pad_z + 1)
        new_height, new_width = height + pad_y * 2, width + pad_x * 2
        new_y, new_x = y - pad_y, x - pad_x

        return CropInfo.create_volume(new_y, new_x, new_height, new_width, new_zs)

    @staticmethod
    def restrict_infos(crop_infos, bounds=None):
        """
        Restrict info to provided volume bounds.
        Args:
            bounds: additional bounds restricting crop_infos

        Returns:
            CropInfo restricted using the provided bounds.
        """
        res = []
        for crop_info in crop_infos:
            restricted = crop_info.restrict(bounds)
            if restricted:
                res.append(restricted)

        return res

    @staticmethod
    def crop_volume(original_volume, crop_infos):
        """
        Re-crop volume from original_data using stored information. Works 3D data.
        Args:
            original_data: HxNxM or HxNxMxC numpy array

        Returns:
            specified volume crop as HxNxM or HxNxMxC numpy array
        """
        slices = []
        for crop_info in crop_infos:
            slices.append(crop_info.crop(original_volume, True))
        return np.array(slices)

    @staticmethod
    def create_volume(y, x, height, width, zs):
        """
        Create list of CropInfo that can form a volume.
        Args:
            y: top of the crop
            x: left of the crop
            height: height of the crop
            width: width of the crop
            zs: list of layers in the volume

        Returns:
            list of CropInfo
        """
        return [CropInfo(y, x, z, (height, width)) for z in zs]

    @staticmethod
    def create_centered_volume(point, shape):
        """
        Create list of CropInfo that can form a volume around a given point.
        Args:
            point: center of the volume (z,y,x)
            shape: shape of the volume (z,y,x)

        Returns:
            list of CropInfo
        """
        assert len(point) == 3, "Point not 3d."
        assert len(shape) == len(point), "Shape do not match point dimensions."
        assert np.prod(shape) > 0, "Empty shape given at point {}".format(point)

        starts = np.array(point, dtype=np.int) - np.array(shape, dtype=np.int) // 2

        return [CropInfo(starts[1], starts[2], z, (shape[1], shape[2])) for z in range(starts[0], starts[0] + shape[0])]

    @staticmethod
    def create_random_volume(available_shape, height, width, z_size):
        """
        Create list of CropInfo that form a random volume inside given shape.
        Args:
            available_shape: S x Y x X in which random
            height: height of the crop
            width: width of the crop
            z_size: number of layers of the crop

        Returns:
            list of CropInfo
        """
        sz, sy, sx = available_shape
        start_z = random.randint(0, sz - z_size)
        start_y = random.randint(0, sy - height)
        start_x = random.randint(0, sx - width)

        return CropInfo.create_volume(start_y, start_x, height, width, range(start_z, start_z + z_size))

    @staticmethod
    def save(crop_infos, file_path):
        """
        Save list of CropInfo into the file as yaml file.
        Args:
            crop_infos: list of CropInfo to save
            file_path: yaml file path
        """

        with open(file_path, "w") as f:
            yaml.dump(crop_infos, f, default_flow_style=False)

    @staticmethod
    def load(file_path):
        """
        Load list of CropInfo from yaml file.
        Args:
            file_path: yaml file path

        Returns:
            list of CropInfo
        """
        with open(file_path, "r") as f:
            crop_infos = yaml.load(f)
            return crop_infos

    def __str__(self):
        return "CropInfo: " + ",".join([str(f) for f in sorted(vars(self).items())])

    def __repr__(self):
        return "CropInfo: " + ",".join([str(f) for f in vars(self).items()])

    @staticmethod
    def is_block(crop_infos):
        """
        Validates that provided crop infos describe a 3d block.
        Args:
            crop_infos: list of CropInfo to check

        Returns:
            True if crop_infos form a block
             or None if list is empty
        """
        if not crop_infos:
            return None

        starts = [crop.zyx_start[1:] for crop in crop_infos]
        shape_y_x = [crop.shape for crop in crop_infos]
        zs = [crop.z for crop in crop_infos]

        return len(set(starts)) == 1 and len(set(shape_y_x)) == 1 and zs == list(range(min(zs), max(zs) + 1))

    @staticmethod
    def block_size(crop_infos):
        """
        Calculate size of the 3d block formed by provided crop infos.
        If it is not a block exception is raised.
        Args:
            crop_infos: list of CropInfo to check

        Returns:
            (z,y,x) shape of the block formed by crop_infos
                or None if list is empty
        """
        if not crop_infos:
            return None

        if not CropInfo.is_block(crop_infos):
            raise ValueError("Crop infos do not form a 3d block.")

        y, x = crop_infos[0].shape
        zs = [crop.z for crop in crop_infos]
        z = max(zs) - min(zs) + 1
        return z, y, x
