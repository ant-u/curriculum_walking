# curriculum-walking

## structure 

project/
│
├── envs/                 # (optional) custom wrappers, env configs
├── scripts/              # training, evaluation scripts
├── models/               # policy architectures, networks
├── configs/              # yaml/json hyperparameter configs
│
└── runs/                 # ⬅ all experiment output goes here


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