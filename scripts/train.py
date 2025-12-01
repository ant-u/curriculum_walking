import json
import os, psutil
import sys
import time
from scripts.util.callbacks import get_all_callbacks
from scripts.util.algorithms import get_PPO
from envs.vec_env import make_env
import yaml
from scripts.view_vec_env import save_gif

PPO_CONFIG = {
    "env_id": "Humanoid-v5",
    "algo": "PPO",
    "policy": "MlpPolicy",
    "device": "cpu",
    "learning_rate": 1e-4,
    "n_steps": 8192,
    "batch_size": 256,
    "clip_range": 0.15,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "gamma": 0.99,  # default
    "gae_lambda": 0.90,
    "max_grad_norm": 0.3,
    "n_epochs": 15,
    "normalize_advantage": True,
    "verbose": 1,
    "tensorboard_log": True,
    
    "timesteps": 10e6,
    "seed": 0,
    "n_envs": 8,
}

CALLBACK_CONFIG = {
    "checkpoint_cb_conf": {
        "save_freq": 1_000_000,
        "save_vecnormalize": True,
        "name_prefix": "ckpt"
    },
    "eval_env_conf": {
        "env_seed": 0,
        "eval_freq": 50_000,
        "deterministic": True,
        "render": False,
    },
    "plot_callback": {
        "window": 100,
        "log_level": 2
    }
}

def main(RUN_DIR):
    print("==== CPU INFO ====")
    print("SLURM_CPUS_ON_NODE:", os.environ.get("SLURM_CPUS_ON_NODE"))
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))

    aff = psutil.Process().cpu_affinity()
    print("CPU affinity:", aff)
    print("Usable cores:", len(aff))

    print("os.cpu_count():", os.cpu_count())
    print("===================")

    env = make_env(n_envs=PPO_CONFIG["n_envs"], seed=PPO_CONFIG["seed"])

    checkpoint_callback, eval_callback, plot_callback = get_all_callbacks(CALLBACK_CONFIG, RUN_DIR)
    
    model = get_PPO(PPO_CONFIG, env, RUN_DIR)
    model.learn(total_timesteps=PPO_CONFIG["timesteps"],
                callback=[checkpoint_callback, eval_callback, plot_callback])
    
    model.save(os.path.join(RUN_DIR, "checkpoints", "last_model"))
    env.save(os.path.join(RUN_DIR, "checkpoints", "vecnormalize_stats.pkl"))
    results_summary = {
        "mean_reward_eval": float(eval_callback.last_mean_reward),
        "n_eval_episodes": eval_callback.n_eval_episodes,
        "timesteps": PPO_CONFIG["timesteps"]
    }

    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
    save_gif(RUN_DIR, display_steps=300)


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


if __name__ == '__main__':
    if len(sys.argv) > 1:  # run_dir path given in call
        run_dir = sys.argv[1]
    else:
        run_dir = make_run_dir(PPO_CONFIG)
    main(run_dir)