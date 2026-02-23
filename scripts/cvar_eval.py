import os
import pickle
from typing import List
import numpy as np
from stable_baselines3 import PPO
from envs.curriculum.curriculum_manager import BufferLevel
from envs.vec_env import load_env
from scripts.view_vec_env import load_configs
from envs.curriculum.level_generator import LevelGenerator
from scripts.util.plot import Plot



def main(model_path, N: int, alpha: float, n_envs: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    PPO_config, env_config, callback_config = load_configs(model_path)
    env_config["terminate_at_x_border"] = 60
    env_config["use_levels"] = True
    env_config["n_envs"] = n_envs
    env = load_env(model_path, env_config)
    model = PPO.load(os.path.join(model_path, "checkpoints", "plain", "best_model"))
    assert env.action_space == model.action_space and \
               env.observation_space == model.observation_space
    
    levels = create_level_list(rng, N)
    descriptions = transform_levels_to_des(levels)
    for lvl_number, (desc, lvl) in enumerate(zip(descriptions, levels)):
        env.env_method("set_level_template", level=desc)
        obs = env.reset()
        done_flags = np.zeros(env.num_envs, dtype=bool)
        all_progress = []
        successfull_runs = 0
        while not all(done_flags):
            action, _ = model.predict(obs, deterministic=True)  # type: ignore  (obs is vec)
            obs, reward, dones, info = env.step(action)
            
            for i, (done, info) in enumerate(zip(dones, info)):
                if done and not done_flags[i]:
                    done_flags[i] = True
                    all_progress.append(info.get("progress", 0))
                    if info.get("success", False):
                        successfull_runs += 1
        lvl.succ_r = successfull_runs / env.num_envs
        print(f"evaluated lvl {lvl_number}, succ_r of {lvl.succ_r}, lvl:{lvl}")
        print(f"  progess: mean:{float(round(np.mean(all_progress), 3))} {[float(round(x, 4)) for x in all_progress]}")
        obs = env.reset()
    levels.sort(key=lambda x: x.succ_r)
    return levels


def create_level_list(rng, N):
    levels = []
    for n in range(N):
        seed = rng.integers(np.iinfo(np.int64).max)  # ~[0, max_int_64]
        params = []
        for min, max in zip(MIN_INIT, MAX_INIT):
            clipped_value = np.clip(rng.uniform(min, max), 0, 1)  # enabling to change probability for 0 levels (or 1)
            params.append(clipped_value)
        bl = BufferLevel(seed, params[0], params[1], params[2], params[3], params[4])
        
        levels.append(bl)
    return levels

def transform_levels_to_des(levels: List[BufferLevel]):
    generator = LevelGenerator()
    descriptions = []
    for bl in levels:
        level = generator.create_level_elements(bl.obstacles, bl.diff_slab, bl.diff_stairs, bl.diff_stump, bl.diff_gap, bl.seed)
        level_des = generator.calculate_element_coords(level)
        descriptions.append(level_des)
    return descriptions


def get_init_buffer_plot(model_path, save_path, N: int, alpha: float, n_envs: int = 10, seed: int = 0):
    levels = main(model_path, N, alpha, n_envs, seed)
    succ_rates_isolated = np.array([level.succ_r for level in levels])
    x = []
    for i in range(0,N+1):
        x.append(i / N)
    y = np.zeros(len(x))
    for i, x_value in enumerate(x):
        y[i] = len(np.where(succ_rates_isolated == x_value)[0])
    plot = Plot(f"Initial Buffer Success rates distribution for {MIN_INIT} to {MAX_INIT}", "Success rate", "Frequency", "Inital buffer with base policy")
    plot.update_with_x(y, x)
    plot.save(os.path.join(save_path, "init_buffer_succ3.svg"))

    with open(os.path.join(save_path, "init_buffer_level_dump3.pkl"), "wb") as f:
        pickle.dump({'levels': levels, "x": x, "y": y, "min": MIN_INIT, "max": MAX_INIT}, f)


# main("runs/base_lidar_gait_height_resistant", 10, 10)
MIN_INIT = [0,0,0,0,0]
MAX_INIT = [0.5, 0.10, 0.10, 0.15, 0.15]
# main("runs/base_lidar_gait_trained_on_obstacles", 10, 10)
get_init_buffer_plot("runs/base_lidar_gait_height_resistant", "runs/base_lidar_gait_height_resistant/eval", 100, 0.1, 20)