# CLB AI

The aim of this project is segment, classify and analyse cells visible in 3D fluorescent imagery with high density of the cells. We use DAPI channel to segment nuclei, classification can use other channels. 
Analysis consist of nuclei statistics as well as spatial statistics between difference cell types. All is wrapped in CLB-Virtum cloud application developed in separate repository.

## Getting code

Due to the usage of submodules, please clone the repo using `ssh` method:

```(bash)
git clone --recursive git@github.com:MicroscopeIT/clb-ai.git
```

or using `https` method:

```(bash)
git clone --recursive https://github.com/MicroscopeIT/clb-ai.git
```

### Dependencies

Project requires `javabridge` library. To install this package Java Development Kit is required. Please install `openjdk-8-jdk` and set `JAVA_HOME` and `javac` path: 

```(bash)
export JAVA_HOME="/usr/lib/jvm/java-1.8.0-openjdk-amd64/"
export PATH=$PATH:/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin
```

## Data storage
Most of the data, results, investigations and annotations are store on GD. See summary document for descriptions: https://docs.google.com/document/d/14sIDNju-_Tvm7jfPodcxYvvdzL0A4jq7Jlll5OHzs9Y

## MicroscopeIT Resources
MicroscopeIT offers its computational resources for this project: VM with GPU onboard.

### Access
VM's address is [cldx-gpu-machine.eastus.cloudapp.azure.com](cldx-gpu-machine.eastus.cloudapp.azure.com). To get the access, contact Olek Cieślak (aleksander.cieslak@microscopeit.com or *olek.cieslak* on Slack).

### General Usage

There is a master virtualenv which should correspond to current `master` setup which should be used. There is also a common place for additional data.
However some portion of the input data (for example all data for segmentation training) is kept in repos `data` folder.

It's preferred that everyone has own workspace directory and works there. Since this VM is equipped with strong GPU, we should make use of it.

If you feel that VM's environment or configuration is missing something, just ping Olek Cieślak and you'll figure something out together.

## Virtum cloud application

There are docker files which pack all the necessary code into a docker which is then used by the backend of Virtum cloud application. The processing are wrapped and defined in `clb/virtules`.


## Continues evaluation

The code is constantly checked in Jenkins (https://ci.azure.microscopeit.com/view/CLB):
- unit tests are run for each commit
- extensive `ce.sh` script is run for each `PR` and after each merge to `master`
    - running full segmentation and its evaluation
    - running full classification and its evaluation
    - running regression retraining of classificators
    - all the results are summarized in description of the build so it is easy to spot any regression
    - all the details of the results are available as artefacts

### Evaluation based on ground truth

The quantitative quality of the solution can be calculated and presented using evaluator. It requires set of inputs/annotations and labels.

Labels can be provided per each crop or by entire dataset. In the latter case the evaluator will first crop the 
annotated regions from full-stack volume using the crop position information provided along with annotations.
However if labels are not present in the folder it can conduct entire segmentation process 
(using provided network model and postprocess parameters) on each of the annotated crops.

Another option is to use evaluator for evaluation of only the postprocessing method. If only probabilities exist in provided folder then 
the `segment_cells` is automatically called to calculate labels prior to evaluation

`python3 -m clb.evaluator --name <friendly_name> --annotated_input <input_folder_with_crops> --annotated_gt <labels_folder_with_crops> --probs <probabilities_folder_or_file> --labels <labels_folder_or_file> --output <output_folder>`

Alternatively we can use datasets parameter to specify which data use to (assuming internal folders `input`/`labels`):

`python3 -m clb.evaluator --name <friendly_name> --data <root_folder_with_crops> --datasets <labels_folder_with_crops> --probs <probabilities_folder_or_file> --labels <labels_folder_or_file> --output <output_folder>`

In order to evaluate labels from IMARIS use:

`python3 -m clb.evaluator --name IMARIS --annotated_input <input_folder_with_crops_and_cropinfo> --annotated_gt <labels_folder_with_crops> --labels <path_to_volume_with_imaris_labels> --output <output_folder>`

#### Parameters

There are flags `--regen_prob --regen_labels` if we want to force regeneration of either probabilities or labels.

The parameters `annotated_input, annotated_gt` can point directly to GD folders (e.g subfolders of CLB_Team/Private/_annotations/T8_S1)

The parameters `probs` is optional as we may not be interested to keep probabilities. 

The IOU threshold for matching objects is configured in `vendor/evaluation.ini` file as `miniousimilarity`.

#### Comparison

The quality of a set of solutions can be compared using vendor/ep/compare.py script, e.g:

`python3 -m vendor.ep.evaluator <output_dir> <output_dir_of_solution_1> <output_dir_of_solution_2>`

#### Viewer

Usually after preparing evaluation we want to investigate all the steps grouped by image. However raw evaluation results are grouped by type not by image.
In order to easy the visual inspection we prepared simple tool: `evalutor_viewer.py` which gathers results from specified folders and copy them into
one folder sorted by the image number. This way we can easily compare: input, probabilities, labels, ground truth, segmentation details. This can be also used
to visually compare various solutions:

`python3 -m clb.evaluator_viewer <evaluation_dir> <output_viewer_dir> <list of path relative to evaluation_dir with images> --rescale`

`evaluation_dir` is the `<output_folder>` from `evaluator`.

Let's assume that there is solution `model_cc`, we can inspect its results using:

`python3 -m clb.evaluator_viewer <evaluation_dir> <output_viewer_dir>
data_input model_cc_prob model_cc_labels data_annotation "model_cc_labels\Output\Segmentation details" --rescale`


## Annotation data preparation

The data for annotation is prepared using multiple tools that can be found in `notebooks/cell_annotations.ipynb`:
- create crops of any sort from raw data
- save its position so it can be used with other tools to recrop
- save voxel size of the data

The data crops used in training and evaluation of segmentation can be found in `data` folder. 
For classification dataset is significantly larger so that only extracted features and test data can be found there (rest is on GD).


## Processes

The main developed features are:
- denoising
- nuclei segmentation
- cell classification
- nuclei statistics
- spatial statistics




### Denoising

There is a Noise2Noise model trained on 512x dataset using DAPI channel available. It can be easily used to reduce noise in the provided imagery.

See `clb.denoising.denoise` for call details.




### Nuclei segmentation

The initial idea for 3D segmentation is to perform 2D segmentation of the Z-slices and then join the resulting segmentation. 
Downside is that we are not using information of Z-axis but it is possible that only X-Y spatial information is enough for a 3D segmentation
of a sufficient quality.


#### Training of the solution 

Training instance segmentation consist of two steps:
- training DCAN neural network `clb.train.train`
- finding hyperparameters for watershed postprocessing `clb.segment.hypertune_segment_cells`

Each important experiment with training should be noted in Google Sheet document.

#### Calculate segmentation

First step is to use DCAN network to calculate probability image:

`python3 -m clb.predict.predict3d --input <input_tiff_path> --output <output_folder> --model <model_path>`

Second step is to convert that probability into label image (threshold and connect pixels into separate objects):

`python3 -m clb.segment.segment_cells --input <input_prob_tiff_path> --output <output_tiff_file> --threshold <threshold eg.0.85> --layer_split_size <layer_split_size eg.17>`

The parameters could also be read from yaml file generated in hyperparametrization process.

The above two steps are combined in one-button instance segmentation solution. Assuming that we have a trained 
model (<model>.h5, <model>.yaml) and best parameters for identify (<model>_ident.yaml) we can run entire process with:

`python3 -m clb.run --input <input_path> --outputs <output_paths> --model <model_path>
[--use_channel <channel>] [--series <series>]`

Input can be:
- multi-page tiff
    - if multi-channel then use `use_channel` to specify which channel is DAPI
- lif
    - use `use_channel` to specify which channel is DAPI
    - use `series` to specify series in file to segment
- uff
    - if multi-channel then use `use_channel` to specify which channel is DAPI

Output can be:
- *.tif (labels channel)
- *.ims (IMARIS file with DAPI and labels channel)
- *.uff (it doesn't have to end with .uff, if output doesn't have extension it
         is assumed to be an .uff file, output file will only have labels 
         channel)
         
There is also possibility to include series name into output path, '{name}' is
treated as a placeholder for series name. If there is no series name in 
metadata (e.g. .tif file) 'series_n' will be used, where n is a series number.

Example command:

`python -m clb.run --input data/series0 --outputs out/{name}.ims out/series0 
--model models/model_7.h5`




### Cell classification

There are two method of cell classification
- based on extracting features from each cell and classification using Random Forests
- based on raw input data (cubes around center of each nuclei) using VGG

#### Training

In order to train classificator we need:
- multi-channel input volumes (first channel is DAPI)
- class annotation (0-negative, 1-positive, 2-uncertain), marked as areas
- instance segmentation model

The training can also be redone starting from various point of the process such as:
- use previously calculated instance segmentation
- use previously calculated features and ground truth cell classes

It is also possible to use manual instance annotation instead of instance segmentation.

The parameters of each training are saved to yaml file so if in doubt check them for existing models.

##### Feature based

The training from scratch can be done using:
`python3 -m clb.classify.train --class_name <compy_with_ground_truth_filenames> --model <output_model_path> --annotated_input <input_folder_path> --annotated_gt <ground_truth_folder_path> {--regen} --channels <channels_from_input_to_use> --instance_model <instance_model_path_to_h5> --labels <folder to store instance labels> --training_data <file to store feature and cell level classes> --results_path <folder to store the evaluation of the classificator>`

Please refer to `clb.classify.train` for more options and details.

##### Cube based

First you need to prepare cubes which can be done using `clb.classify.prepare_train` by specifing:
- cell_crop_size=16 (size of the cube in um)

The training can be run using `clb.classify.nn.train` script so please refer to that file for details.

#### Volume classification

In order to calculate classes for entire volume of imagery we need:
- multi-channel input volume
- trained instance segmentation model
- trained classificator model

The classification can be done using:

`python3 -m clb.classify.classify --input <input_filepath> --channels <channels>
--outputs <output_filepath_with_class_probs> --model <classificator_pkl_path> 
--instance-model <instance_segmentation_model_h5_path> 
--labels <path_to_labels> --discrete <discretization>`

`channels` parameter is very important and it should provide the same markers are used in training.

It is important to note that you can specify any of the two types of models defined above. All the important parameters are read from the corresponding yaml file.

All input and output types can be the same as in case of segmentation. It is vital that the 

Argument `--discrete` allows to choose discretization type for output. Possible
options are `binary` and `4bins`. With first option two groups are returned
(yes/no). With second option four groups are returned (yes/weak yes/weak no/no).
Without this option raw probabilities are returned. In case of .uff output
`binary` option should be used.

Example command:

`python3 -m clb.classify.classify --input data/series0 --channels 0,1 --outputs out/series0 out/{name}.tif 
--model models/classification/model_7.pkl --instance-model models/model_7.h5 --labels out/{name}_labels.tif 
--discrete binary`
 



### Nuclei statistics

The cells in the imagery can be analysed by calculation of the statistics for each of the nuclei using `clb.stats.all_stats`. It generates information about:
- statistics per cell type in entire volume
- morphology of each nuclei
- marker intensity of each nuclei

Additional option it to generate fixed set of plots based on the above statistics using `clb.stats.scatterplots`.


### Spatial statistics

The position and concentration of the cells can be analysed using `clb.stats.spatial_stats`:
- it will generate homogenity graph which can show if cells of given types are attracted or repelled by reference cell type
