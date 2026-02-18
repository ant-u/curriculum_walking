import argparse
import json
import os, psutil
import time
from datetime import timedelta
import time
from scripts.util.callbacks import get_all_callbacks
from scripts.util.algorithms import get_PPO, load_PPO
from envs.vec_env import load_env, make_env
from envs.config import PPO_CONFIG, ENV_CONFIG, CURR_CONFIG, CALLBACK_CONFIG
import yaml




def main(RUN_DIR: str, train_on_path: str, message: str):
    """main function for training. run_dir is (new) folder for saving the trained model. 
    train_on is path to already trained model for continuing training"""
    print_cpu_info()
    dump_premature_summary(message, RUN_DIR)
    ENV_CONFIG["render_lidar"] = False  # For training, rendering is irelevant
    
    if train_on_path == None:
        env = make_env(ENV_CONFIG)
        model = get_PPO(PPO_CONFIG, env, RUN_DIR)
    else:
        env = load_env(train_on_path, ENV_CONFIG)
        model = load_PPO(PPO_CONFIG, env, train_on_path, RUN_DIR)
        assert env.action_space == model.action_space and \
               env.observation_space == model.observation_space

    # env.env_method("set_env_level_slab", x_ratio=0.8, height=0.1)
    checkpoint_cb, eval_cb, plot_cb, curr_cb = get_all_callbacks(
        CALLBACK_CONFIG, ENV_CONFIG, CURR_CONFIG, RUN_DIR, train_on_path)
    
    start_time = time.monotonic()
    model.learn(total_timesteps=PPO_CONFIG["timesteps"], callback=[checkpoint_cb, *eval_cb, plot_cb, curr_cb])
    end_time = time.monotonic()

    model.save(os.path.join(RUN_DIR, "checkpoints", "last_model"))
    env.save(os.path.join(RUN_DIR, "checkpoints", "last_vecnormalize_stats.pkl"))
    time_diff = timedelta(seconds=end_time - start_time)
    dump_summary(eval_cb, PPO_CONFIG["timesteps"], model, message, time_diff, train_on_path, RUN_DIR)


def dump_premature_summary(message, RUN_DIR):
    """Dump short summary with message so it doesn't get lost when training is killed."""
    summary_short = {"message": message}
    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(summary_short, f, indent=4)


def dump_summary(eval_cb, timesteps, model, message, timedelta, train_on_path, RUN_DIR):
    """Dump long summary after training finished."""
    results_summary = {
        "mean_reward_eval": float(eval_cb[0].last_mean_reward),
        "n_eval_episodes": eval_cb[0].n_eval_episodes,
        "timesteps": timesteps,
        "obs_shape": model.observation_space.shape,
        "message": message,
        "running_time": str(timedelta)
    }
    if train_on_path != None:
       results_summary.update({"based_on": train_on_path})
    with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=4)


def make_run_dir(ppo_cnfg, env_cnfg, curr_cnfg, callback_cnfg):
    """Make directory for new training. Includes subdirectories and also saves snapshot of config."""
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
            {'CURR_CONFIG': curr_cnfg},
            {'CALLBACK_CONFIG': callback_cnfg},
        ]
        yaml.add_representer(list, represent_list)
        yaml.dump_all(cnfg_list, f, indent=4)
    return run_dir

def represent_list(dumper, data):
    if any(isinstance(item, dict) for item in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

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
        run_dir = make_run_dir(PPO_CONFIG, ENV_CONFIG, CURR_CONFIG, CALLBACK_CONFIG)
    main(run_dir, args.train, args.message)