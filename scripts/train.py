import os
import json
import yaml
import time
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# ============================================================
# 🔧 1) CONFIGURATION (Edit here or load via YAML)
# ============================================================

CONFIG = {
    "env_id": "Humanoid-v5",
    "algo": "PPO",
    "seed": 0,
    "n_envs": 8,
    "timesteps": 3_000_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tensorboard_log": True
}

# ============================================================
# 📁 2) Create RUN directory following our structure
# ============================================================

def make_run_dir(cfg):
    folder_name = f"{cfg['env_id'].lower()}_{cfg['algo'].lower()}_lr{cfg['learning_rate']:.0e}_seed{cfg['seed']}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{folder_name}_{timestamp}")
    
    subdirs = ["checkpoints", "logs", "videos", "configs"]
    for sd in subdirs:
        os.makedirs(os.path.join(run_dir, sd), exist_ok=True)

    # save config copy
    with open(os.path.join(run_dir, "configs", "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f)

    return run_dir

RUN_DIR = make_run_dir(CONFIG)

# ============================================================
# 🧱 3) Environment creation
# ============================================================

def make_env(env_id):
    def _make():
        env = gym.make(env_id)
        env = Monitor(env)  # record returns/ep lengths
        return env
    return _make

def create_vec_env(env_id, n_envs):
    if n_envs > 1:
        env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_id)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    return env

env = create_vec_env(CONFIG["env_id"], CONFIG["n_envs"])

# ============================================================
# 🎯 4) Callback configuration
# ============================================================

CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoints")
LOG_PATH = os.path.join(RUN_DIR, "logs")

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=CHECKPOINT_PATH,
    name_prefix="ckpt"
)

eval_env = create_vec_env(CONFIG["env_id"], 1)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=CHECKPOINT_PATH,
    log_path=LOG_PATH,
    eval_freq=50_000,
    deterministic=True,
    render=False,
)

# ============================================================
# 🤖 5) Create PPO Agent
# ============================================================

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=CONFIG["learning_rate"],
    gamma=CONFIG["gamma"],
    verbose=1,
    tensorboard_log=(LOG_PATH if CONFIG["tensorboard_log"] else None),
    seed=CONFIG["seed"]
)

# ============================================================
# 🚀 6) Train
# ============================================================

model.learn(
    total_timesteps=CONFIG["timesteps"],
    callback=[checkpoint_callback, eval_callback]
)

# ============================================================
# 💾 7) SAVE artifacts
# ============================================================

# save final model
model.save(os.path.join(RUN_DIR, "checkpoints", "last_model"))

# save normalization stats
env.save(os.path.join(RUN_DIR, "checkpoints", "vecnormalize_stats"))

# save evaluation results summary
results_summary = {
    "mean_reward_eval": float(eval_callback.last_mean_reward),
    "n_eval_episodes": eval_callback.n_eval_episodes,
    "timesteps": CONFIG["timesteps"]
}

with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
    json.dump(results_summary, f, indent=4)