import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from gymnasium.wrappers import TimeLimit
from envs.humanoid_base import HumanoidEnvBase
from envs.humanoid_curr import HumanoidEnvCurr
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv as HumanoidEnvDefault  # NOTE: renaming to default for less confusion


def make_env_default(n_envs: int = 1, seed: int = 0):
    """Generates the default gymnasium humnanoid_v5 env with no extras whatsoever."""
    vec_env = make_vec_env(HumanoidEnvDefault, n_envs=n_envs, seed=seed, monitor_dir=None)
    norm_env = VecMonitor(vec_env)
    norm_env = VecNormalize(norm_env, norm_reward=True, clip_reward=10, norm_obs=True)
    return norm_env


def make_env_base(n_envs: int = 1, seed: int = 0):
    """Generates an env with envs.humanoid_v5, which is NOT the original humanoid_v5, but extended."""
    vec_env = make_vec_env(HumanoidEnvBase, n_envs=n_envs, seed=seed, monitor_dir=None)
    norm_env = VecMonitor(vec_env)
    norm_env = VecNormalize(norm_env, norm_reward=True, clip_reward=10, norm_obs=True)
    return norm_env


def make_env_curr(n_envs: int = 1, max_steps: int = 1000, seed: int = 0, xml_file_name = None):
    env_kwargs = {}
    if xml_file_name is not None:
        env_kwargs = {"xml_file": xml_file_name}
    
    vec_env = make_vec_env(
        HumanoidEnvCurr, 
        n_envs=n_envs, 
        seed=seed, 
        wrapper_class=TimeLimit, 
        wrapper_kwargs={"max_episode_steps": max_steps},
        env_kwargs=env_kwargs)
    
    norm_env = VecNormalize(vec_env, norm_reward=True, clip_reward=10, norm_obs=True)
    return norm_env

def laod_env_curr(path: str, n_envs: int = 1, seed: int = 0, xml_file_name = None):
    joined_path = os.path.join(path, "checkpoints", "vecnormalize_stats.pkl")
    if xml_file_name is None:
        vec_env = make_vec_env(HumanoidEnvCurr, n_envs=n_envs, seed=seed, monitor_dir=None)
    else:
        vec_env = make_vec_env(HumanoidEnvCurr, n_envs=n_envs, seed=seed, monitor_dir=None, env_kwargs={"xml_file": xml_file_name})
    mon_env = VecMonitor(vec_env)
    norm_env = VecNormalize.load(joined_path, mon_env)  # Load VecNormalize statistics into this new VecEnv
    return norm_env
