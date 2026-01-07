import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from gymnasium.wrappers import TimeLimit
from envs.humanoid_base import HumanoidEnvBase
from envs.humanoid_curr import HumanoidEnvCurr
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv as HumanoidEnvDefault  # NOTE: renaming to default for less confusion


def make_env(cnfg):
    assert cnfg["env_id"] in ["HumanoidEnvDefault", "HumanoidEnvBase", "HumanoidEnvCurr"],\
    "env_id has to be HumanoidEnvDefault, HumanoidEnvBase or HumanoidEnvCurr"
    match cnfg["env_id"].lower():  # no case sensitivity
        case "humanoidenvdefault":
            return make_env_default(cnfg)
        case "humanoidenvbase":
            return make_env_base(cnfg)
        case "humanoidenvcurr":
            return make_env_curr(cnfg)


def make_env_default(cnfg):
    """Generates the default gymnasium humnanoid_v5 env with no extras whatsoever."""
    env_kwargs = None
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        path = os.path.abspath(os.path.join("models", cnfg["xml_file"]))
        env_kwargs = {"xml_file": path}
    vec_env = make_vec_env(HumanoidEnvDefault, n_envs=cnfg["n_envs"], seed=cnfg["seed"], monitor_dir=None,
                           env_kwargs=env_kwargs)
    norm_env = VecNormalize(vec_env, clip_reward=cnfg["clip_reward"],
                            norm_reward=cnfg["norm_reward"], norm_obs=cnfg["norm_obs"])
    return norm_env


def make_env_base(cnfg):
    """Generates an env with envs.humanoid_v5, which is NOT the original humanoid_v5, but extended."""
    env_kwargs = None
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        env_kwargs = {"xml_file": cnfg["xml_file"]}
    vec_env = make_vec_env(HumanoidEnvBase, n_envs=cnfg["n_envs"], seed=cnfg["seed"], monitor_dir=None,
                           env_kwargs=env_kwargs)
    norm_env = VecNormalize(vec_env, clip_reward=cnfg["clip_reward"],
                            norm_reward=cnfg["norm_reward"], norm_obs=cnfg["norm_obs"])
    return norm_env


def make_env_curr(cnfg):
    env_kwargs = {"cnfg": cnfg}  # "render_mode": "human"
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        env_kwargs.update({"xml_file": cnfg["xml_file"]})
    if cnfg["max_steps"] > 0:
        wrapper_class = TimeLimit
        wrapper_kwargs = {"max_episode_steps": cnfg["max_steps"]}
    
    vec_env = make_vec_env(
        HumanoidEnvCurr, 
        n_envs=cnfg["n_envs"], 
        seed=cnfg["seed"], 
        wrapper_class=wrapper_class, 
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs=env_kwargs)
    
    norm_env = VecNormalize(vec_env, clip_reward=cnfg["clip_reward"],
                            norm_reward=cnfg["norm_reward"], norm_obs=cnfg["norm_obs"])
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


def load_render_env(stats_path: str, cnfg, render_mode: str = "human"):
    env = make_vec_env(HumanoidEnvCurr, n_envs=1, seed=cnfg["seed"], 
                       env_kwargs={"render_mode": render_mode, "cnfg": cnfg})
    env = VecNormalize.load(stats_path, env)  # Load VecNormalize statistics into this new VecEnv
    env.training = False        # disables running stats updates
    env.norm_reward = False     # do not normalize rewards during inference
    return env
