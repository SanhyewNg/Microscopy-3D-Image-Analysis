import glob
import os
from functools import lru_cache

from clb.cropping import CropInfo

LABELS_SUFIX = '_segmented'
CLASSES_SUFIX = '_classes'


class ClassVolumeSet:
    def __init__(self, input, crop_info, gt, label, classes):
        """
        Prepare set of paths which represent: input, annotation, labels, classes for one annotated volume.
        Args:
            input: path to file with input image
            gt: path to file with annotation
            crop_info: path to yaml file with crop info which may include the annotation method
            label: path to file with labels
            classes: path to file with cell level class prediction
        """
        self.input = input
        self.gt = gt
        self.crop_info = crop_info
        self.label = label
        self.label_exist = label is not None and os.path.isfile(label)
        self.classes = classes
        self.cell_classes = {}
        self.cell_features = {}
        self.cell_crops = {}
        self.cell_classes_predicted = {}
        self.cell_positions = {}

    def load_crop_info(self):
        if self.crop_info is None or not os.path.isfile(self.crop_info):
            return None
        else:
            return CropInfo.load(self.crop_info)

    @property
    def voxel_size(self):
        crop_infos = self.load_crop_info()
        if crop_infos is not None:
            return crop_infos[0].voxel_size
        return None

    @property
    def crop_name(self):
        return os.path.basename(self.input)

    @property
    @lru_cache(1)
    def merged_cell_data(self):
        res = {}
        for id, class_data in self.cell_classes.items():
            cell_data = class_data.copy()
            cell_data.update(self.cell_features[id])
            res[id] = cell_data
        return res

    def __str__(self):
        sb = []
        for key in ['input', 'crop_info', 'gt', 'label']:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)


def find_all_tiffs(folder_path, class_name=None):
    tiff_paths = sorted(glob.glob(os.path.join(folder_path, '*.tif')))
    if class_name:
        # First try to get files with exactly class_name.
        exact_tiff_paths = [p for p in tiff_paths if "_" + class_name + "_" in os.path.basename(p)]
        if not exact_tiff_paths:
            # Try get files with class_name as a prefix.
            prefix_tiff_paths = [p for p in tiff_paths if "_" + class_name in os.path.basename(p)]
        return exact_tiff_paths or prefix_tiff_paths
    return tiff_paths


def get_all_multiple_datasets(root_path, names, input_folder_sufix, ground_truth_folder_sufix, labels, classes,
                              class_name=None):
    all_data = []
    for dataset in names:
        dataset_path = os.path.join(root_path, dataset)
        all_data += get_all_datasets(os.path.join(dataset_path, input_folder_sufix),
                                     os.path.join(dataset_path, ground_truth_folder_sufix),
                                     labels, classes, class_name)
    return all_data


def validate_pairs_match(input_paths, annotation_paths):
    if len(input_paths) < len(annotation_paths):
        raise Exception("More ground truth files than input files.")
    else:
        if len(input_paths) > len(annotation_paths):
            print("More input files than ground truth files (maybe some are still not annotated?).")

            for input_path, annotation_path in zip(input_paths, annotation_paths):
                input_name = os.path.splitext(os.path.basename(input_path))[0]
                annotation_name = os.path.splitext(os.path.basename(annotation_path))[0]
                #  We usually add prefix or suffix to input name.
                if not input_name in annotation_name:
                    raise Exception("Two matched files have incompatible names: ", input_name, annotation_name)
            print("Inputs without annotations:", len(input_paths) - len(annotation_paths))
            print("Not matched input files:")
            for k in range(len(annotation_paths), len(input_paths)):
                input_name = os.path.splitext(os.path.basename(input_paths[k]))[0]
                print("\t", input_name)


def get_all_datasets(input, ground_truth, labels, classes, class_name=None):
    os.makedirs(input, exist_ok=True)
    os.makedirs(ground_truth, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    
    input_files = find_all_tiffs(input)
    gt_files = find_all_tiffs(ground_truth, class_name)

    validate_pairs_match(input_files, gt_files)

    res = []
    for input, gt in zip(input_files, gt_files):
        file_name = os.path.basename(input)
        file_name_without_extension = file_name[:-4]
        input_path_without_extension = input[:-4]

        expected_annotation_info = input_path_without_extension + ".yaml"
        expected_label = os.path.join(labels, file_name_without_extension + LABELS_SUFIX + ".tif")
        if classes:
            expected_classes = os.path.join(classes, file_name_without_extension + CLASSES_SUFIX + ".tif")
        else:
            expected_classes = None
        dataset = ClassVolumeSet(input, expected_annotation_info, gt, expected_label, expected_classes)
        res.append(dataset)
    return res


def update_with_prefix(self, items_to_add, prefix_to_add, exclude_prefix=tuple(["id"])):
    """
    Extends self by the data from dictionary with keys in form of prefix_to_add+key.
    Args:
        self: dictionary to update
        items_to_add: dictionary to add to self
        prefix_to_add: prefix prepended to each key from items_to_add
        exclude_prefix:
            keys in cell_features not to add prefix
    Returns:
        dictionary extended by dictionary data
    """
    if prefix_to_add != '':
        prefix_to_add += '_'

    for k, v in items_to_add.items():
        if k not in exclude_prefix:
            self[prefix_to_add + k] = v
        else:
            self[k] = v


def add_data_with_prefix(self, data_to_add, new_data_prefix):
    """
    Add features data to self with provided prefix.
    Args:
        self: dictionary from cell id to feature dictionary
        data_to_add: dictionary from cell id to feature dictionary
        new_data_prefix: prefix prepended to each feature copied to self

    Returns:
        dictionary from cell id to feature dictionary updated with data_to_add
    """
    assert self == {} or sorted(data_to_add.keys()) == sorted(self.keys())

    keys = self.keys() if self != {} else data_to_add.keys()
    for id in keys:
        if id not in self:
            self[id] = {}
        update_with_prefix(self[id], data_to_add[id], new_data_prefix)


def save_to_tsv(df, path):
    """
    Save DataFrame to file in a readable way.
    Args:
        df: DataFrame to save
        path: tsv path
    """
    df.to_csv(path, columns=sorted(df.columns), index=True, sep='\t')

def group_by_image(x, y=None):
    """Return mapping of image. Required for cross-validation
    
    Args:
        x: Dataframe with MultiIndex where first index is image name.
        y: Series with MultiIndex where first index is image name. (Ignored kept for compatibility)
    """
    return [idx[0] for idx in x.index]