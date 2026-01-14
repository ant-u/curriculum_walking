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
    vec_env = make_vec_env(HumanoidEnvDefault, n_envs=cnfg["n_envs"], seed=cnfg["seed"],
                           env_kwargs=env_kwargs)
    norm_env = VecNormalize(vec_env, clip_reward=cnfg["clip_reward"],
                            norm_reward=cnfg["norm_reward"], norm_obs=cnfg["norm_obs"])
    return norm_env


def make_env_base(cnfg):
    """Generates an env with envs.humanoid_v5, which is NOT the original humanoid_v5, but extended."""
    env_kwargs = None
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        path = os.path.abspath(os.path.join("models", cnfg["xml_file"]))
        env_kwargs = {"xml_file": path}
    vec_env = make_vec_env(HumanoidEnvBase, n_envs=cnfg["n_envs"], seed=cnfg["seed"],
                           env_kwargs=env_kwargs)
    norm_env = VecNormalize(vec_env, clip_reward=cnfg["clip_reward"],
                            norm_reward=cnfg["norm_reward"], norm_obs=cnfg["norm_obs"])
    return norm_env


def make_env_curr(cnfg):
    env_kwargs = {"cnfg": cnfg}  # "render_mode": "human"
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        env_kwargs.update({"xml_file": cnfg["xml_file"]})
    if "env_kwargs" in cnfg and cnfg["env_kwargs"] != {}:
        env_kwargs.update(cnfg["env_kwargs"])
    wrapper_class = None
    wrapper_kwargs = None
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


def load_env(path: str, cnfg):
    """Load env as specified by config. picks env type from config, 
    please make sure obs space and act space are same size as in previous one.
    Also loads per default the 'last_vecnormalize_stats'."""
    env_kwargs = None
    if cnfg["xml_file"] != None and cnfg["xml_file"] != "":
        xml_path = os.path.abspath(os.path.join("models", cnfg["xml_file"]))
        env_kwargs = {"xml_file": xml_path}

    previous_env_id = path.split('/')[1].split("_")[0]
    assert previous_env_id.lower() == cnfg["env_id"].lower(), \
        "Previous env and new env differ! Use same env for continuing training."
    
    match cnfg["env_id"].lower():
        case "humanoidenvdefault":
            env_class = HumanoidEnvDefault
        case "humanoidenvbase":
            env_class = HumanoidEnvBase
        case "humanoidenvcurr":
            env_class = HumanoidEnvCurr
            env_kwargs = {"cnfg": cnfg}

    if "env_kwargs" in cnfg and cnfg["env_kwargs"] != {}:
        env_kwargs.update(cnfg["env_kwargs"])

    joined_path = os.path.join(path, "checkpoints", "last_vecnormalize_stats.pkl")
    vec_env = make_vec_env(env_class, n_envs=cnfg["n_envs"], seed=cnfg["seed"], env_kwargs=env_kwargs)
    norm_env = VecNormalize.load(joined_path, vec_env)  # Load VecNormalize statistics into this new VecEnv
    return norm_env


def load_render_env(stats_path: str, cnfg, render_mode: str = "human"):
    env_class = HumanoidEnvCurr
    env_kwargs={"render_mode": render_mode, "cnfg": cnfg}
    if "env_id" in cnfg and cnfg["env_id"].lower() != "humanoidenvcurr":
        match cnfg["env_id"].lower():  # no case sensitivity
            case "humanoidenvdefault":
                env_class = HumanoidEnvDefault
                env_kwargs = {"render_mode": render_mode}
            case "humanoidenvbase":
                env_class = HumanoidEnvBase
                env_kwargs = {"render_mode": render_mode}
        if cnfg["xml_file"] != '':  # Catching case of using default xml file (in config as '')
            env_kwargs["xml_file"] = cnfg["xml_file"]
    if "env_kwargs" in cnfg and cnfg["env_kwargs"] != {}:
        env_kwargs.update(cnfg["env_kwargs"])

    env = make_vec_env(env_class, n_envs=1, seed=cnfg["seed"], 
                       env_kwargs=env_kwargs)
    env = VecNormalize.load(stats_path, env)  # Load VecNormalize statistics into this new VecEnv
    env.training = False        # disables running stats updates
    env.norm_reward = False     # do not normalize rewards during inference
    return env
