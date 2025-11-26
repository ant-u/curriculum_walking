import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

def train():
    # env = gym.make("Humanoid-v5")
    env = make_env('Humanoid-v5', seed=0, n_envs=8)

    model = PPO("MlpPolicy", env, verbose=1, device='cpu', learning_rate=0.0003)
    model.learn(total_timesteps=2_000_000)

    model.save("alternatives/humanoid_ppo2m")
    env.close()


def make_env(env_id: str, seed: int = 0, n_envs: int = 4):
    # make_vec_env will create vectorized copies; each copy is wrapped with Monitor automatically
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed, monitor_dir=None)
    # stable-baselines3 recommends VecMonitor (keeps episode rewards/lengths)
    vec_env = VecMonitor(vec_env)
    return vec_env

if __name__ == '__main__':
    train()