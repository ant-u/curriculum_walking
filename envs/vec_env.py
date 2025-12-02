from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from envs.humanoid_v5 import HumanoidEnv
from envs.humanoid_v5_hmap import HumanoidEnvHmap


def make_env(n_envs: int = 1, seed: int = 0):
    vec_env = make_vec_env(HumanoidEnv, n_envs=n_envs, seed=seed, monitor_dir=None)
    norm_env = VecMonitor(vec_env)
    norm_env = VecNormalize(norm_env, norm_reward=True, clip_reward=10, norm_obs=True)
    return norm_env


def make_env_hmap(n_envs: int = 1, seed: int = 0):
    vec_env = make_vec_env(HumanoidEnvHmap, n_envs=n_envs, seed=seed, monitor_dir=None)
    norm_env = VecMonitor(vec_env)
    norm_env = VecNormalize(norm_env, norm_reward=True, clip_reward=10, norm_obs=True)
    return norm_env
