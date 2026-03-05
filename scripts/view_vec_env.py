import argparse
import numpy as np
from stable_baselines3 import PPO
import imageio
import os
import yaml
from envs.vec_env import load_render_env
from scripts.train import PPO_CONFIG, ENV_CONFIG, CALLBACK_CONFIG
from envs.curriculum.level_generator import LevelGenerator


def view_vec_env(run_dir: str, display_loop: int = 100, display_steps: int = 2500, export_gif: bool = False) -> list:
    """View vectorized env. run_dir neeeds /checkpoints and /videos.
    - display_loop gives how many resets are done.
    - display_steps gives how many steps per episode are rendered.
    """
    PPO_config, env_config, callback_config = load_configs(run_dir)
    render_mode = "human" if not export_gif else "rgb_array"
    if os.path.exists(os.path.join(run_dir, "checkpoints", "best_model.zip")):
        base_path = os.path.join(run_dir, "checkpoints")
    else:
        base_path = os.path.join(run_dir, "checkpoints", "general")
    model = PPO.load(os.path.join(base_path, "best_model.zip"))
    stats_path = os.path.join(base_path, "best_vecnormalize_stats.pkl")

    cam_config = {
        "trackbodyid": 1,
        "distance": 45.0,
        "lookat": np.array((30.0, 0.0, 2.0)),
        "elevation": -10.0,
    }
    env = load_render_env(stats_path, env_config, render_mode, width=1800, height=900, terminate_when_unhealthy=True,
                          default_camera_config=cam_config)
    gen = LevelGenerator()
    eval_env_number = 4
    params = CALLBACK_CONFIG["eval_env_conf"]["eval_levels"][eval_env_number]["params"]
    seed = CALLBACK_CONFIG["eval_env_conf"]["eval_levels"][eval_env_number]["seed"]
    level_elems = gen.create_level_elements(*params, seed)
    # level_elems = gen.create_level_elements(0.84765, 0.18758, 0.95449, 0.22743, 0.35667, 5287405870979402264)
    # level_elems = gen.create_level_elements(0.78038, 0.19690, 0.12998, 0.36041, 0.75572, 6221207345909872665)
    level_elems = gen.create_level_elements(1, 0.19690, 0.12998, 0.36041, 0.99, 6221207345909872665)

    level_des = gen.calculate_element_coords(level_elems)

    env.env_method("set_level_template", level=level_des)
    env.venv.envs[0].env.terminate_on_x = 60

    frames = []
    obs = env.reset()
    done = False
    total_steps = []
    all_progress = []
    display_loop = 1 if export_gif else display_loop
    for i in range(display_loop):
        frames = []
        for step in range(display_steps):
            action, _ = model.predict(obs, deterministic=True)  # type: ignore  (obs is vec)
            obs, reward, done, info = env.step(action)
            
            if export_gif:
                frame = env.render()
                frames.append(frame)
            if done and 'progress' in info[0].keys():
                all_progress.append(float(round(info[0]['progress'], 5)))
            if done:
                total_steps.append(step)
                break
        obs = env.reset()
    env.close()
    print("total env steps: " + str(total_steps))
    print(np.mean(total_steps))
    print(f"total progress: {all_progress}")
    print(np.mean(all_progress))
    print(f"mean success {len(np.where(np.array(all_progress) >= 1.0)[0])}")
    print(f"best run: {max(all_progress)}")
    print(f"used level nr {eval_env_number}: {CALLBACK_CONFIG["eval_env_conf"]["eval_levels"][eval_env_number]["name"]}")
    return frames


def load_configs(run_dir: str):
    path = os.path.join(run_dir, "configs", "config_used.yaml")
    with open(path, 'r') as f:
        doc = yaml.safe_load_all(f)
        configs = list(doc)
    if len(configs) == 3:
        PPO_config = configs[0]['PPO_CONFIG']
        env_config = configs[1]['ENV_CONFIG']
        callback_config = configs[2]['CALLBACK_CONFIG']
    elif len(configs) == 4:
        PPO_config = configs[0]['PPO_CONFIG']
        env_config = configs[1]['ENV_CONFIG']
        curr_config = configs[2]['CURR_CONFIG']
        callback_config = configs[3]['CALLBACK_CONFIG']
    else:
        PPO_config = PPO_CONFIG
        env_config = ENV_CONFIG
        callback_config = CALLBACK_CONFIG
    return PPO_config, env_config, callback_config
    
        
def save_gif(run_dir: str, display_steps: int = 300):
    frames = view_vec_env(run_dir, display_loop=1, display_steps=display_steps, export_gif=True)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    OUTPUT_GIF = os.path.join(video_dir, "gait_humanoid.gif")
    
    FPS = 30
    imageio.mimsave(OUTPUT_GIF, frames, fps=FPS)
    print("Saved GIF to", OUTPUT_GIF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View vec env')
    parser.add_argument('path', type=str, help='Path to run dir')
    parser.add_argument('-g', '--gif', action='store_true', help='Export as GIF')
    
    args = parser.parse_args()
    if not args.gif:
        view_vec_env(args.path)
    else:
        save_gif(args.path)