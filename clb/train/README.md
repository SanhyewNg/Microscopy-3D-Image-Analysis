# Networks - training, tuning and predicting

## Training

To train the network (either locally or in the cloud) the thing that the
training couldn't run without is the **data**. In our project it is assumed,
that all the data is located on our `git-lfs` infrastructure. Okay, but what
kind of data should be there? Input images can be of any size, but most of
the networks were trained on `200x200` grayscale images (meaning, they have
a single channel with values from 0 to 255), that were then upscaled to
`256x256` (since this is the default input size for the network). To train,
however, we also need ground-truth. Here, ground-truths are assumed to be
grayscale images that have each object labeled uniquelly with chosen label
value. Background is marked as `'0'` and all other objects are marked as
`2`, `3`, `4`, ..., `100` etc. Note, however, that value `1` is reserved for
areas that we call **blobs**, meaning they were too hard to be annotated and
thus forming a uniform region of (most probably) touching cells.

### Locally

To train the network locally, there are no special steps you need to
perform. Ensure the data is in `data/` directory (`data` should be located
in main repo directory). Please note, that in current infrastracture all the
data we have is located in `git-lfs`, making it extremely easy to train the
network on different setups and keep the consistency of the data between the
experiments. To run the training use the following command scheme:

```
python -m clb.train.train
    required:
    --architecture [either 'dcan' or 'unet']
    --epochs [number of epochs]
    --batch [batch size]
    --augments [number of augments]
    --trim_method ["resize" | "padding" | "reflect"]
    --train_data [train data directory]
    --val_data [validation data directory]
    --model [output path for model checkpoint file]


    optional:
    --channels [declaring how many channels your input has. For now,
                Multi-channel supported only by unet]
    --lr [learning rate value]
    --im_dim [network input image size]
    --bnd_weight [how strongly boundaries loss influences validation loss]
    --obj_weight [how strongly objects loss influences validation loss]
    --dropout [value for dropout]
    --dcan_final_act [final activation function in DCAN architecture]
    --tb_dir [path to directory to output TensorBoard logs]
    --csv_log [path to output CSV file with training logs]
    --seed [value to seed all randomness]
    --args_load_path [path to .yaml file to read parameters from]
```

Be mindful to use absolute paths to the data given.

#### Runing TensorBoard locally

When training the network, you would probably like to see what's going on
during the process (not after it, but that's the case too). That is the
reason TensorBoard was created - to let you do:

* previewing of the metrics live during the training,
* previewing sample predictions from validation dataset after each epoch,
* previewing your network's graph

Note that during the training you can pass `--tb_dir` CLI parameter that
points to the output directory for logs that can be read by TensorBoard.
Default is `'tensorboard/'`, so when training the network you can simply
run:
```
tensorboard --logdir='tensorboard'
```

And the TensorBoard will be started on `localhost:6006` - that's the default
address for this app. You should then open it in your web browser to preview
the results. It is, however, not that simple when training in the cloud.
More details later in this README.


## Hyperparameter tuning

One functionality that was developed some time ago (and is still functional,
however to limited extent) is hyperparameter tuning, coded in `hypertune.py`
script. It provides some basic functionalities to find best batch size and
best learning rate, assuming some searching scheme. The inspiration for
learning rate search was the following note from cs231n course that can be
found [here](http://cs231n.github.io/neural-networks-3/#hyper).

Sample command to run learning rate tuning:

```
python -m clb.train.hypertune tune_lr
    --architecture [network architecture, either 'dcan' or 'unet']
    --batch [fixed batch size]
    --train_data ["path to training data"]
    --val_data ["path to validation data"]
```

Sample command to run batch size tuning:

```
python -m clb.train.hypertune tune_batch
    --architecture [network architecture, either 'dcan' or 'unet']
    --epochs [for how many epochs will each training be run]
    --train_data ["path to training data"]
    --val_data ["path to validation data"]
```

**Note that this functionality is quite imperfect and what is being currently
  done is rather a manual search for many hyperparameters. The idea, however,
  is quite nice and could be used in the future after several improvements.
  What is more, this functionality works only locally, i.e. it doesn't make
  use of the potential of BatchAI service.**

## Predicting

There's a script called `predict.py` that contains the code that can be used
to generate probability maps of cells and edges which then constitute input
for segmentation (both semantic and instance).

**Generating predictions shouldn't be mislead with the evaluation.**

To run this script, use the following command (you have to be in main repo
directory, not `clb`, `vendor` or any other):

```
python -m clb.train.predict
    required:
    --architecture [dcan or unet]
    --channels [declaring how many channels your input has. For now,
                Multi-channel supported only by unet]
    --config [yaml file with configuration dumped during network training]
    --test_data [directory with test data to run predict on]

    optional:
    --plot [preview the network predictions (for comparison and insight)]
    --output [output directory to save predictions and heatmaps to (if not
              provided, not output results will be saved)]
```

For instance:

```
python -m clb.train.predict
    --config models/model_6_dreamy_jackson.yaml
    --test_data data/training/T8/test/
    --plot
```

Note that instead of passing direct path to the model, you have to pass yaml
config file. It contains several values that are used by prediction script
that you shouldn't hardcode or pass by CLI. Sample configuration file looks
like this:

```yaml
{architecture: unet, epochs: 500, batch: 6, augments: 10, trim_method: resize,
 train_data: data/training/T8/train, val_data: data/training/T8/val,
 model: models/model_6_dreamy_jackson.h5, lr: 0.001, im_dim: 256,
 bnd_weight: 0.5, obj_weight: 0.5, dropout: 0.0, dcan_final_act: tanh,
 tb_dir: 'tensorboard/', csv_log: 'log.csv', seed: 48}
```
