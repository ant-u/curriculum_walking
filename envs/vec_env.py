from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.humanoid_v5 import HumanoidEnv

def train():
    # env = gym.make("Humanoid-v5")
    env = make_env(n_envs=8, seed=0)

    model = PPO("MlpPolicy", env, verbose=1, device='cpu', learning_rate=0.0003)
    model.learn(total_timesteps=2_000_000)

    model.save("alternatives/humanoid_ppo2m")
    env.close()


def make_env(n_envs: int = 1, seed: int = 0):
    vec_env = make_vec_env(HumanoidEnv, n_envs=n_envs, seed=seed, monitor_dir=None)
    norm_env = VecNormalize(vec_env, norm_reward=True, clip_reward=10, norm_obs=True)
    norm_env = VecMonitor(norm_env)
    return norm_env

if __name__ == '__main__':
    train()