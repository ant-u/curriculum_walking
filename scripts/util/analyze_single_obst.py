import os
import pickle
from matplotlib import pyplot as plt
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
    render_mode = None
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
    env_config["healthy_z_range"] = [-10, 10]
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
    tries = 100

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



def create_plot(data_path, save_path, position="lower left"):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    x = data["diff"]*100
    y = [[],[],[],[],[],[]]
    for i, run in enumerate(data["runs"]):
        y[i] = [100*np.mean(x) for x in run]

    font_size_axis_ticks = 12
    font_size_title = 12
    font_size_label = 12
    font_size_legend = 12

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel("Obstacle difficulty (in %)", fontsize=font_size_label)
    ax1.set_ylabel("Success rate (in %)", fontsize=font_size_label)  # color=color_reward

    colors = [(0.5, 1, 0.5), (1, 0.5, 0.5), (0.5, 0.5, 1), (0.5, 1, 1)]

    line1, = ax1.plot(x, y[0], color=colors[0], linewidth=2, label="Slab")
    line2, = ax1.plot(x, y[4], color=colors[0], linewidth=2, linestyle="--", label="Slab (desc.)")  #linestyle="--"
    line3, = ax1.plot(x, y[1], color=colors[1], linewidth=2, label="Stairs")  
    line4, = ax1.plot(x, y[5], color=colors[1], linewidth=2, linestyle="--", label="Stairs (desc.)")  #linestyle="--"
    line5, = ax1.plot(x, y[2], color=colors[2], linewidth=2, label="Stump")  
    line6, = ax1.plot(x, y[3], color=colors[3], linewidth=2, label="Gap")

    ax1.tick_params(axis="y", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.set_ylim(bottom=0, top=100)
    ax1.set_xlim(left=0, right=100)
    ax1.set_xticks(ticks=x[::2], labels=[int(x) for x in x[::2]])


    lines = [line1, line2, line3, line4, line5, line6]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=position, fontsize=font_size_legend)
    # plt.title(f'Buffer Evolution', fontsize=font_size_title) #, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")

# view_vec_env("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350", "runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/eval/single_obstacle_results.pkl")
# view_vec_env("runs/base_lidar_gait_height_resistant", "runs/base_lidar_gait_height_resistant/eval/single_obstacle_results_z_range.pkl")
create_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/eval/single_obstacle_results.pkl", "thesis_plots/experiments/plots/exp_f_single_obst.pdf")
create_plot("runs/base_lidar_gait_height_resistant/eval/single_obstacle_results_z_range.pkl", "thesis_plots/base_policy/base_policy_single_obst.pdf", "upper right")