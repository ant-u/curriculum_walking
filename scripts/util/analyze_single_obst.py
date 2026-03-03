import os
import pickle
import numpy as np
from stable_baselines3 import PPO
from envs.config import CALLBACK_CONFIG
from envs.curriculum.level_generator import ElementType, Level, LevelGenerator
from envs.vec_env import load_render_env
from scripts.view_vec_env import load_configs


def view_vec_env(run_dir: str, save_path, display_steps: int = 2500) -> list:
    """View vectorized env. run_dir neeeds /checkpoints and /videos.
    - display_loop gives how many resets are done.
    - display_steps gives how many steps per episode are rendered.
    """
    PPO_config, env_config, callback_config = load_configs(run_dir)
    render_mode = "human"
    if os.path.exists(os.path.join(run_dir, "checkpoints", "best_model.zip")):
        base_path = os.path.join(run_dir, "checkpoints")
    else:
        # base_path = os.path.join(run_dir, "checkpoints")
        base_path = os.path.join(run_dir, "checkpoints", "plain")
    # model = PPO.load(os.path.join(base_path, "ckpt_297270000_steps.zip"))
    model = PPO.load(os.path.join(base_path, "best_model.zip"))
    # stats_path = os.path.join(base_path, "best_vecnormalize_297270000_steps.pkl")
    stats_path = os.path.join(base_path, "best_vecnormalize_stats.pkl")

    cam_config = {
        "trackbodyid": 1,
        "distance": 45.0,
        "lookat": np.array((30.0, 0.0, 2.0)),
        "elevation": -10.0,
    }
    env = load_render_env(stats_path, env_config, render_mode, width=1800, height=900, terminate_when_unhealthy=True,
                          default_camera_config=cam_config)
    env.venv.envs[0].env.terminate_on_x = 15
    gen = LevelGenerator()
    gen.rng = np.random.default_rng(0)
    level_elems = Level()

    runs = [[],[],[],[],[],[]]

    obstacle_types = [ElementType.SLAB, ElementType.STAIRS, ElementType.STUMP, ElementType.GAP, ElementType.SLAB, ElementType.STAIRS]
    flips = [-10, -10, -10, -10, 10, 10]
    difficulties = np.arange(21) * 5 / 100
    tries = 1

    for i, type in enumerate(obstacle_types):
        print(f"analyzing {type} with ({flips[i]})")
        for diff in difficulties:
            diff_results = []

            element = gen.get_element_params(type, diff, diff, diff, diff, flips[i])
            element.pos = [5, 6, 7, 8, 9, 10]
            level_elems = Level()
            if element.n >= 2:
                for x in range(element.n-1):
                    new_pos = np.array([11, 12, 13]) + 3*x
                    element.pos.extend(new_pos)
            level_elems.elements.append(element)
            level_des = gen.calculate_element_coords(level_elems)
            env.env_method("set_level_template", level=level_des)

            obs = env.reset()
            done = False
            for n in range(tries):
                for step in range(display_steps):

                    action, _ = model.predict(obs, deterministic=True)  # type: ignore  (obs is vec)
                    obs, reward, done, info = env.step(action)
                    
                    if done and 'success' in info[0].keys():
                        diff_results.append(int(info[0]['success']))
                    if done:
                        break
                obs = env.reset()
            runs[i].append(diff_results)
            print(f"using diff {diff:.2f} succ rate: {np.mean(diff_results)}")
    env.close()
    with open(save_path, "wb") as f:
        pickle.dump({"runs": runs, "diff": difficulties, "types": ["Slab", "Stairs", "Stump", "Gap", "neg Slab", "neg Stairs"]}, f)


# view_vec_env("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350", "runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/eval/single_obstacle_results.pkl")
view_vec_env("runs/base_lidar_gait_height_resistant", "runs/base_lidar_gait_height_resistant/eval/single_obstacle_results.pkl")