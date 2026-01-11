import argparse
import json
import os, psutil
import time
from datetime import timedelta
import time
from scripts.util.callbacks import get_all_callbacks
from scripts.util.algorithms import get_PPO, load_PPO
from envs.vec_env import laod_env_curr, make_env
import yaml

PPO_CONFIG = {
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
    "timesteps": 20e6,
    "seed": 0,
    "partition": "Krater",  # e.g. NvidiaAll or Krater
}

ENV_CONFIG = {
    "xml_file": "humanoid.xml",  # NOTE: full name of a file in ./models. If empty / none, defaults are used
    "env_id": "HumanoidEnvCurr",  # "HumanoidEnvDefault", "HumanoidEnvBase", "HumanoidEnvCurr"
    "n_envs": 8,
    "max_steps": 0,  # 0 disables max steps
    "use_lidar": False,
    "render_lidar": False,  # NOTE: is disabled during training
    "use_relative_height": False,
    "seed": 0,
    "n_points_x": 6,
    "n_points_y": 5,
    "y_width": 1.5,
    "x_forward": 4,
    "x_start": -1,
    "norm_reward": True,
    "norm_obs": True,
    "clip_reward": 10,  # default: 10
}

CALLBACK_CONFIG = {
    "checkpoint_cb_conf": {
        "save_freq": 3_000_000,
        "save_vecnormalize": True,
        "name_prefix": "ckpt"
    },
    "eval_env_conf": {
        "env_seed": 0,
        "eval_freq": 500_000,
        "deterministic": True,
        "render": False,
    },
    "plot_callback": {
        "window": 100,
        "log_level": 2
    }
}

def main(RUN_DIR, train_on, message):
    """main function for training. run_dir is (new) folder for saving the trained model. 
    train_on is path to already trained model for continuing training"""
    print_cpu_info()
    ENV_CONFIG["render_lidar"] = False  # For training, rendering is irelevant
    summary_short = {"message": message}
    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(summary_short, f, indent=4)

    if train_on == None:
        env = make_env(ENV_CONFIG)
        model = get_PPO(PPO_CONFIG, env, RUN_DIR)
    else:
        env = laod_env_curr(train_on, n_envs=PPO_CONFIG["n_envs"], seed=PPO_CONFIG["seed"], xml_file_name=PPO_CONFIG["xml_file"])
        model = load_PPO(PPO_CONFIG, env, train_on, RUN_DIR)
        print(f"using pretrained model from: {train_on}")

    # env.env_method("set_env_level_slab", x_ratio=0.8, height=0.1)

    checkpoint_cb, eval_cb, plot_cb, curr_cb = get_all_callbacks(CALLBACK_CONFIG, ENV_CONFIG, RUN_DIR)
    
    start_time = time.monotonic()
    model.learn(total_timesteps=PPO_CONFIG["timesteps"], callback=[checkpoint_cb, eval_cb, plot_cb, curr_cb])
    end_time = time.monotonic()

    model.save(os.path.join(RUN_DIR, "checkpoints", "last_model"))
    env.save(os.path.join(RUN_DIR, "checkpoints", "last_vecnormalize_stats.pkl"))
    results_summary = {
        "mean_reward_eval": float(eval_cb.last_mean_reward),
        "n_eval_episodes": eval_cb.n_eval_episodes,
        "timesteps": PPO_CONFIG["timesteps"],
        "obs_shape": model.observation_space.shape,
        "message": message,
        "running_time": str(timedelta(seconds=end_time - start_time))
    }
    if train_on != None:
       results_summary.update({"based_on": train_on})
    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
    # save_gif(RUN_DIR, display_steps=300)


def make_run_dir(ppo_cnfg, env_cnfg, callback_cnfg):
    folder_name = f"{env_cnfg['env_id'].lower()}_{ppo_cnfg['algo'].lower()}_lr{ppo_cnfg['learning_rate']:.0e}_seed{ppo_cnfg['seed']}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{folder_name}_{timestamp}")
    
    subdirs = ["checkpoints", "logs", "videos", "configs"]
    for sd in subdirs:
        os.makedirs(os.path.join(run_dir, sd), exist_ok=True)

    # save config copy
    with open(os.path.join(run_dir, "configs", "config_used.yaml"), "w") as f:
        cnfg_list = [
            {'PPO_CONFIG': ppo_cnfg},
            {'ENV_CONFIG': env_cnfg},
            {'CALLBACK_CONFIG': callback_cnfg},
        ]
        yaml.dump_all(cnfg_list, f, indent=4)
    return run_dir


def print_cpu_info():
    """Printing info about cpus available to stdout, mostly important for slurm training."""
    print("==== CPU INFO ====")
    print("SLURM_CPUS_ON_NODE:", os.environ.get("SLURM_CPUS_ON_NODE"))
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))

    aff = psutil.Process().cpu_affinity()
    print("CPU affinity:", aff)
    print("Usable cores:", len(aff))

    print("os.cpu_count():", os.cpu_count())
    print("===================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a policy')
    parser.add_argument('-p', '--path', type=str, required=False, help='Path for already created run_dir')
    parser.add_argument('-t', '--train', type=str, required=False, help='Path for already trained policy for further training')
    parser.add_argument('-m', '--message', type=str, required=False, help='Comment on training')
    args = parser.parse_args()
    if args.path != None:  # run_dir path given in call
        run_dir = args.path
    else:
        run_dir = make_run_dir(PPO_CONFIG, ENV_CONFIG, CALLBACK_CONFIG)
    main(run_dir, args.train, args.message)