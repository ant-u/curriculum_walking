import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import imageio
import os
import torch
from stable_baselines3 import PPO
from envs.humanoid_v5 import HumanoidEnv  # your custom wrapper


def load_render_env(stats_path: str, seed: int = 0, render_mode: str = "human"):
    env = make_vec_env(HumanoidEnv, n_envs=1, 
                       seed=seed, env_kwargs={"render_mode": render_mode})
    env = VecNormalize.load(stats_path, env)  # Load VecNormalize statistics into this new VecEnv

    env.training = False        # disables running stats updates
    env.norm_reward = False     # do not normalize rewards during inference
    return env

def view_vec_env(run_dir: str, display_loop: int = 10, display_steps: int = 500, export_gif: bool = False) -> list:
    """View vectorized env. run_dir neeeds /checkpoints and /videos.
    - display_loop gives how many resets are done.
    - display_steps gives how many steps per episode are rendered.
    """
    render_mode = "human" if not export_gif else "rgb_array"
    model = PPO.load(os.path.join(run_dir, "checkpoints", "last_model"))
    env = load_render_env(os.path.join(run_dir, "checkpoints", "vecnormalize_stats.pkl"), render_mode=render_mode)
    
    frames = []
    obs = env.reset()
    done = False
    for i in range(display_loop):
        frames = []
        for step in range(display_steps):
            action, _ = model.predict(obs, deterministic=True)  # type: ignore  (obs is vec)
            obs, reward, done, info = env.step(action)
            
            frame = env.render()
            frames.append(frame)
            if done:
                obs = env.reset()
    env.close()
    return frames
    
        
def save_gif(run_dir: str, display_steps: int = 300):
    frames = view_vec_env(run_dir, display_loop=1, display_steps=display_steps, export_gif=True)
    OUTPUT_GIF = os.path.join(run_dir, "videos", "cassie_walk.gif")
    FPS = 30
    imageio.mimsave(OUTPUT_GIF, frames, fps=FPS)
    print("Saved GIF to", OUTPUT_GIF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View vec env')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to run dir')
    args = parser.parse_args()
    view_vec_env(args.path)