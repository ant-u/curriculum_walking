import os
import pickle
from typing import List
import numpy as np
from stable_baselines3 import PPO
from envs.curriculum.curriculum_manager import BufferLevel
from envs.vec_env import load_env
from scripts.view_vec_env import load_configs
from envs.curriculum.level_generator import LevelGenerator
from scripts.util.plot import Plot, LogPlot, DoubleLogPlot

class EvalLevel(BufferLevel):
    def __init__(self, seed, obstacles, diff_slab, diff_stairs, diff_stump, diff_gap,
                 regret=None, succ_r=None, learnability=None, progress=None):
        super().__init__(seed, obstacles, diff_slab, diff_stairs, diff_stump, diff_gap, regret, succ_r, learnability)
        self.progress = progress



def main(path, ppo_path, vec_env_path, N: int, alpha: float, n_envs: int = 10, seed: int = 0, levels=None):
    rng = np.random.default_rng(seed)
    PPO_config, env_config, callback_config = load_configs(path)
    env_config["terminate_at_x_border"] = 60
    env_config["use_levels"] = True
    env_config["n_envs"] = n_envs
    env = load_env(path, env_config, vec_norm_path=vec_env_path)
    model = PPO.load(os.path.join(path, ppo_path))
    assert env.action_space == model.action_space and \
               env.observation_space == model.observation_space
    
    if levels is None:
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
        lvl.progress = np.mean(all_progress)
        print(f"evaluated lvl {lvl_number}, succ_r of {lvl.succ_r}, lvl:{lvl}")
        print(f"  progess: mean:{float(round(np.mean(all_progress), 3))} {[float(round(x, 4)) for x in all_progress]}")
        obs = env.reset()
    levels.sort(key=lambda x: x.succ_r)
    return levels


def create_level_list(rng, N) -> List[EvalLevel]:
    levels = []
    for n in range(N):
        seed = rng.integers(np.iinfo(np.int64).max)  # ~[0, max_int_64]
        params = []
        for min, max in zip(MIN_INIT, MAX_INIT):
            clipped_value = np.clip(rng.uniform(min, max), 0, 1)  # enabling to change probability for 0 levels (or 1)
            params.append(clipped_value)
        bl = EvalLevel(seed, params[0], params[1], params[2], params[3], params[4])
        
        levels.append(bl)
    return levels

def transform_levels_to_des(levels: List[EvalLevel]):
    generator = LevelGenerator()
    descriptions = []
    for bl in levels:
        level = generator.create_level_elements(bl.obstacles, bl.diff_slab, bl.diff_stairs, bl.diff_stump, bl.diff_gap, bl.seed)
        level_des = generator.calculate_element_coords(level)
        descriptions.append(level_des)
    return descriptions


def get_init_buffer_plot(path, ppo_path, vec_env_path, save_path, N: int, alpha: float, n_envs: int = 10, seed: int = 0):
    levels = main(path, ppo_path, vec_env_path, N, alpha, n_envs, seed)
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


def cvar_eval(path, ppo_path, vec_env_path, save_path, N: int, alphas: float, n_envs: int = 10, seed: int = 0, levels=None):
    levels = main(path, ppo_path, vec_env_path, N, alphas, n_envs, seed, levels=levels)
    prog_levels = levels.copy()
    levels.sort(key=lambda x: x.succ_r)
    prog_levels.sort(key=lambda x: x.progress)

    x = np.array(alphas) / 100
    y_succ_r = np.zeros(len(x))
    y_prog = np.zeros(len(x))
    for i, alpha in enumerate(x):
        n_elems = int(alpha * len(levels))
        elems_succ = levels[:n_elems]
        elems_prog = prog_levels[:n_elems]
        mean_succ_r = None
        mean_prog = None
        if len(elems_succ) > 0:
            mean_succ_r = np.mean([elem.succ_r for elem in elems_succ])
        if len(elems_prog) > 0:
            mean_prog = np.mean([elem.progress for elem in elems_prog])
        y_succ_r[i] = mean_succ_r
        y_prog[i] = mean_prog

    plot = DoubleLogPlot(f"CVaR evaluation with N={N} and {n_envs} episodes for {ppo_path.split("/")[-1]}", "alpha", "Success rate", "Success rate", "Average Progress")
    plot.update_with_x(y_succ_r, y_prog, x)
    plot.save(os.path.join(save_path, f"{ppo_path.split("/")[-1]}_cvar_evaluation.svg"))
    with open(os.path.join(save_path, f"{ppo_path.split("/")[-1]}_cvar_buffer_dump.pkl"), "wb") as f:
        pickle.dump({'levels': levels, 'x': x, 'y_succ': y_succ_r, 'y_prog': y_prog}, f)

def cvar_with_predefined_levels(path, ppo_path, vec_env_path, save_path, level_path, N: int, alphas: float, n_envs: int = 10, seed: int = 0):
    with open(level_path, "rb") as f:
        data = pickle.load(f)
    levels = data['levels']
    for lvl in levels:
        lvl.succ_r = None
    cvar_eval(path, ppo_path, vec_env_path, save_path, N, alphas, n_envs, seed, levels=levels)


# main("runs/base_lidar_gait_height_resistant", 10, 10)
MIN_INIT = [0,0,0,0,0]
MAX_INIT = [1,1,1,1,1]
# MAX_INIT = [0.5, 0.10, 0.10, 0.15, 0.15]
# cvar_eval("runs/base_lidar_gait_height_resistant", "runs/base_lidar_gait_height_resistant/eval", 1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# -------- exp A: ---------------
# cvar_with_predefined_levels("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125",
#                             "checkpoints/ckpt_45864000_steps", "checkpoints/ckpt_vecnormalize_45864000_steps.pkl", 
#                             "runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)  
# cvar_with_predefined_levels("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125",
#                             "checkpoints/ckpt_160008000_steps", "checkpoints/ckpt_vecnormalize_160008000_steps.pkl", 
#                             "runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/eval/2nd_try_160M",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)  

# -------- exp B: ---------------
# cvar_with_predefined_levels("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206",
#                             "checkpoints/ckpt_217368000_steps", "checkpoints/ckpt_vecnormalize_217368000_steps.pkl", 
#                             "runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# get_init_buffer_plot("runs/base_lidar_gait_height_resistant", "runs/base_lidar_gait_height_resistant/eval", 100, 0.1, 20)

# -------- exp c: ---------------
# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231",
#                             "checkpoints/ckpt_12210000_steps.zip", "checkpoints/ckpt_vecnormalize_12210000_steps.pkl", 
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231/eval/",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)  
# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231",
#                             "checkpoints/ckpt_280950000_steps.zip", "checkpoints/ckpt_vecnormalize_280950000_steps.pkl", 
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231/eval/",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)  

# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-154549", 
#                             "checkpoints/ckpt_273360000_steps.zip", "checkpoints/ckpt_vecnormalize_273360000_steps.pkl", 
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-154549/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-232323",
#                             "checkpoints/ckpt_244710000_steps.zip", "checkpoints/ckpt_vecnormalize_244710000_steps.pkl", 
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-232323/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-160853",
#                             "checkpoints/ckpt_39870000_steps.zip", "checkpoints/ckpt_vecnormalize_39870000_steps.pkl", 
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-160853/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)  

# cvar_with_predefined_levels("runs/result_exp_c/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154150",
#                             "checkpoints/ckpt_38184000_steps.zip", "checkpoints/ckpt_vecnormalize_38184000_steps.pkl",
#                             "runs/result_exp_c/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154150/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# -------- exp D: ---------------
# cvar_with_predefined_levels("runs/result_exp_d/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100512",
#                             "checkpoints/ckpt_28110000_steps.zip", "checkpoints/ckpt_vecnormalize_28110000_steps.pkl",
#                             "runs/result_exp_d/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100512/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_d/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161329",
#                             "checkpoints/easy/best_model.zip", "checkpoints/easy/best_vecnormalize_stats.pkl", 
#                             "runs/result_exp_d/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161329/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_d/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154623",
#                             "checkpoints/easy/best_model.zip", "checkpoints/easy/best_vecnormalize_stats.pkl",
#                             "runs/result_exp_d/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154623/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# -------- exp E: ---------------
# cvar_with_predefined_levels("runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618",
#                             "checkpoints/ckpt_19650000_steps.zip", "checkpoints/ckpt_vecnormalize_19650000_steps.pkl",
#                             "runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)
# cvar_with_predefined_levels("runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618",
#                             "checkpoints/easy/best_model.zip", "checkpoints/easy/best_vecnormalize_stats.pkl",
#                             "runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

cvar_with_predefined_levels("runs/result_exp_e/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155353",
                            "checkpoints/ckpt_37920000_steps.zip", "checkpoints/ckpt_vecnormalize_37920000_steps.pkl", 
                            "runs/result_exp_e/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155353/eval",
                            "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
                            1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)
# cvar_with_predefined_levels("runs/result_exp_e/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155353",
#                             "checkpoints/easy/best_model.zip", "checkpoints/easy/best_vecnormalize_stats.pkl", 
#                             "runs/result_exp_e/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155353/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# # -------- exp F: ---------------
# cvar_with_predefined_levels("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350",
#                             "checkpoints/ckpt_297270000_steps.zip", "checkpoints/ckpt_vecnormalize_297270000_steps.pkl",
#                             "runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)
# cvar_with_predefined_levels("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350",
#                             "checkpoints/ckpt_192960000_steps.zip", "checkpoints/ckpt_vecnormalize_192960000_steps.pkl",
#                             "runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120621",
#                             "checkpoints/easy/best_model.zip", "checkpoints/easy/best_vecnormalize_stats.pkl",
#                             "runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120621/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_f/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161918",
#                             "checkpoints/ckpt_50610000_steps.zip", "checkpoints/ckpt_vecnormalize_50610000_steps.pkl", 
#                             "runs/result_exp_f/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161918/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

# cvar_with_predefined_levels("runs/result_exp_f/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155601",
#                             "checkpoints/ckpt_49464000_steps.zip", "checkpoints/ckpt_vecnormalize_49464000_steps.pkl", 
#                             "runs/result_exp_f/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155601/eval",
#                             "runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl",
#                             1000, [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 100], 10)

