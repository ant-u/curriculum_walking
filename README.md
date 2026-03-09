# curriculum-walking

Master Thesis implementation by Anton Utz.

This project implements a obstacle-based RL environment for humanoid-v5 gym (farama).

## structure 

```
project/
│
├── envs/                 # environments for training, humanoid_curr is the env for curriculum obstacle learning
├── models/               # xml mujoco files for different environments
├── scripts/              # training, evaluation scripts
│
└── runs/                 # ⬅ all experiment output goes here
```

```
runs/
│
├── humanoid_v5_ppo_seed0/
│   ├── checkpoints/      # saved model weights (.zip)
│   ├── logs/             # tensorboard + metrics
│   ├── videos/           # rollouts, gifs, mp4s
│   ├── configs/          # copy of the used hyperparameters yaml
│   └── results.json      # final metrics summary
│
├── humanoid_v5_ppo_seed1/
```

## instalation

use the [pyproject.toml](./pyproject.toml) file. No guarantee for completeness.

```
python -m pip install .
```

## usage

### train

The project is build with python modules. To run a script with dependicies on other parts on the project, the python module usage is required:

```
python -m scripts.train
```

In [scripts/train.py](scripts/train.py) the training is handled. The command above starts a training.
Configuration is loaded from [config.py](envs/config.py) at training start.
For special train usage, such as loading a pretrained policy, the flag -t (--train, for train on) is required.
To initialize a training with the base policy from the thesis, use this command:

```
python -m scripts.train -t runs/base_lidar_gait_height_resistant
```

For furhter description of all parameters, refer to [scripts/train.py](scripts/train.py).

### view policy


