# curriculum-walking

Master Thesis implementation by Anton Utz.

This project implements a obstacle-based RL environment for humanoid-v5 gym (farama).

## structure 

```
project/
│
├── envs/                 # environments for training, humanoid_curr is the env for curriculum obstacle learning
|   └── curriculum/       # logic for the curriculum contained here
├── models/               # xml mujoco files for different environments
├── scripts/              # training, evaluation scripts
|   └── util/             # different utility files, e.g. callbacks
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

To view a policy in the environment it has been trained on, the [scripts/view_vec_env.py](scripts/view_vec_env.py)
can be used. It automatically loads the config of a viewed policy from the config snapshot, ensuring consistent usage.
The script requires as first positional argument the path to the training run, like:

```
python -m scripts.view_vec_env runs/base_lidar_gait_height_resistant
```

With an optional ```-g``` flag, a gif is exported into the videos folder of the run.
The policy checkpoint loaded by view_vec_env is with first priority [checkpoints/best_model.zip](checkpoints/best_model.zip), and if such a file does not exist, [checkpoints/general/best_model.zip](checkpoints/general/best_model.zip). 
In the same location, the respective best_vecnormalize_stats.pkl needs to be placed.
For all experiments conducted for the thesis, a [general](general) folder has been created. It contains the best version of the training run, as described in the thesis.

For new training runs, this 'best version' would have to be identified. Tools are the graphs in the logs/ directory of each run, as well as [scripts/cvar_eval.py](scripts/cvar_eval.py).
A best practice for this framework is to identify the best checkpoint of a training run (possibly the last), create a 'general' directory in the checkpoints, and copy the best checkpoint there, whiile also renaming it. 

The script also implements a small statistic tracking of how good the viewed policy is able to solve the viewed environment. The results are printed after the last episode. 
To set a certain level to view the policy on, use the settings in the beginning of view_vec_env() in the file.
An evaluation level can be loaded from the configuration by changing ```eval_env_number```.
0 is the flat level, 1-4 are slabs, stairs, stumps and gaps, and 5-8 are easy, medium, hard and extreme.
An own, different custom level can be set by not loading the environment with the arguments from the config 
(line 36-38), but with own arguments.

## Custom Humanoid Environment

The source for the level-based env is [envs/humanoid_curr.py](envs/humanoid_curr.py). It is based on Humanoid-v5 gym by farama. Since some adaptions had to be made to this parent environment, a custom version of the parent environment is located at [envs/humanoid_base.py](envs/humanoid_base.py). The difference to the original version is not big, only an 
exclusion flag for the absolute z_height is added, as well as a more specific creation of the observation space from the .xml file. The original version simpyl uses the length of arrays as mujoco qpos, without checking if any parts in it do not belong to the robot.

humanoid_curr needs to be called with config file, handing over the custom xml path, however, overwrittes the one from config. Through **kwargs usage, arguments for the parent environment humanoid-v5 can be passed. Possibly interesting for flags as ```terminate_when_unhealthy```. 

Creation of the environment is in this work handled through [envs/vec_env.py](envs/vec_env.py), which also wrapps the environment with the vectorization wrapper. [envs/vec_env.py](envs/vec_env.py) also is used for loading environments, either for continued training or to view the policy in them. The ```make_env``` function can create three differenct environments, as given in the config in ```env_id```: HumanoidEnvDefault, HumanoidEnvBase or HumanoidEnvCurr.
The default version is the original humanoid-v5 gym, base is the adapted humanoid-v5 parent version with exclusion of z_height, and Curr is the level-based curriculum environment used in this work (which is based on HumanoidEnvBase).
These different modes are primarily used for debugging and comparison of the custom env to the default version.

Through the usage of different .xml files used in the environments, a different mujoco setup is used. [models/humanoid.xml](models/humanoid.xml) is the default version with an infinite plane. [models/humanoid_hmap.xml](models/humanoid_hmap.xml) is an adaption of this default plane-based setup, which includes the points for height map measurement rendering. [models/humanoid_curr.xml](models/humanoid_curr.xml) is the obstacle-based version with walls and movable cuboids, primarily used in this work.

## Curriculum logic

In [train.py](scripts/train.py), callbacks are created through [scripts/util/callbacks.py](scripts/util/callbacks.py). One of these callbacks is the curriculum callback, which implements the high-level logic for the curriculum. This mainly contains the logic of 
which rollout buffer is used, a smaller one for simple evaluation or the default one for actual training. The callback instanciates
the [envs/curriculum/curriculum_manager.py](envs/curriculum/curriculum_manager.py), which is managing the curriculum process. It contains the buffer, picks the new levels and decides whether evalutaion or training is done. It has an instance of [envs/curriculum/level_generator.py](envs/curriculum/level_generator.py), used for sampling random levels.
