# Denoising

## General

We are using encoder-decoder architecture with skip connections in this approach. 
All trained models accepted inputs with shape 128,128, so this is default parameter 
in many places. All models were trained on 8-bit greyscale images. For now accepted 
inputs and outputs are only .png files, images are stored internally as numpy 
arrays with dtype=np.float32. 

## Inference

To denoise image use following command:

```
python -m clb.denoising.denoise --input <input_path> --output <output_path>
--model <model_path> [--patches-shape] <y,x> [--patches-stride <y,x>]

```

Description of arguments:
* `--input`, path to input file, for now only .png files are supported.
* `--output`, path to save output file, it should also be .png.
* `--model`, path to model.
* `--patches-shape`, shape of patches used during denoising of bigger image,
                     it should fit network input shape, should be in form of
                     y,x,1 (without whitespace) defaults to 128,128,1.
* `--patches-stride`, stride used during denoising of bigger image, should be
                      in form of y,x,1 (without whitespace). Defaults to 
                      32,32,1.
                      
## Training

IMPORTANT !!!

In case of arguments which are pattern it's necessary to quote pattern to avoid
bash glob expansion. Data should be organised in a way where images than can be paired
as input, target should be in the same directory. 
                                         
To train new model use following command:

```
python -m clb.denoising.train non_artificial --train-fovs <train_pattern> 
--val_fovs <val_pattern> --batch-size <size> --epochs <epochs_num> 
--learning-rate <lr> --save-dir <results_path> [--augment <aug_flag>]
[--noise2noise <n2n_flag>] [--stddev <stddev>] [--shuffle <shuffle_flag>]
[--seed <seed>] [--image-save-interval <img_save_interval>] 
[--model-save-interval <model_save_interval>] [--args-load-path <load_path>]
[--args-save-path <save_path>]
```  

Description of arguments:
* `--train-fovs-pattern`, global expression for directories with images groups.
* `--val-fovs-pattern`, global expression for directories with images groups, validation set.
* `--batch-size`, size of batch.
* `--epochs`, number of epochs.
* `--learning-rate`, learning rate (it's not changing during training for now).
* `--save-dir`, main directory to save results of training, see docstring of
                'prepare_directory_tree' function in 'clb/denosing/train.py'
                file for more info about structure of this directory.
* `--augment`, should data be augmented, defaults to True.
* `--seed`, seed for randomness, defaults to 3.
* `--model-save-frequency`, interval (in epochs) between two saves of 
                           checkpoints.
