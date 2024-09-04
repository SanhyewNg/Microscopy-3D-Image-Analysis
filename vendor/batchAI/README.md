# Code for batchAI functionality

## Important

To be able to send jobs on CLB batchAI cluster, `batchai.yaml` file is
necessary. It contains confidential data, so contact either Olek Cieślak or
Bartek Miselis to get it directly from them.

## Creating batchAI cluster

**BatchAICluser already exists for CLB.**
If, however, you would like to create a new one, please contact Olek Cieślak
or Bartek Miselis in order to get proper bash script / command for it.

## Queuing experiment to be run on cluster

To queue an experiment on BatchAI cluster, use the following command scheme:

```
python vendor/BatchAI/train.py run
    --job_name [name of BatchAI job to be created]
    --command ["command to execute on BatchAI machine"]
    --branch "master"
```

`--job_name` can be any name you want, but it would be good to stick with
the notation from the document with all the experiments that can be found 
[here](https://docs.google.com/spreadsheets/d/1CaRd6EoSzuxicqbgrvO4s5eJiZzwMHhiEkBlBS-cxWA/edit#gid=0).

`--command` should be a training command, e.g.:

```
python3.6 -m clb.train.train 
    --batch 6
    --lr 0.001
    --epochs 250
    --augments 30 
    --model objective_kilhs.h5 
    --train_data 'data/training/T3/train+data/training/T5/train'
    --val_data 'data/training/T3/val+data/training/T5/val'
    --trim_method resize
    --dropout 0.0"
```

**More details on BatchAI can be found in clb/train/README.md file.**

## Mounting fileshare to access outputs from experiments

Please note that virtual fileshare is **extremely** slow. Prefered solution
is to use cldx-batchai-proxy virtual machine that is located in eastus.
Mount fileshare there, run TensorBoard there, scp from this machine and
preview outputs' directory structure there. You'll find it more comfortable,
trust me.

```(bash)
python vendor/batchAI/train.py mount
```

## Running tensorboard

**Fileshare has to be mounted first!**

```(bash)
python vendor/batchAI/train.py tensorboard --jobs [job_id]
```

`job_id` can be retrieved either from command `python
vendor/batchAI/train.py run ...` (very bottom of it's output) or directly
from accessing Azure portal.
