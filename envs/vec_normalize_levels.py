from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Optional
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from envs.humanoid_curr import HumanoidEnvCurr


class VecNormalizeLevels(VecNormalize):
    def __init__(
        self,
        env_cnfg: dict,
        # venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        norm_obs_keys: Optional[list[str]] = None,
    ):
        self.env_cnfg = env_cnfg
        venv = self.make_venvs(self.env_cnfg)
        super().__init__(venv, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon, norm_obs_keys)

    def change_level(self, xml_file_name):
        env_cnfg_to_pass = self.env_cnfg.copy()
        env_cnfg_to_pass["xml_file"] = xml_file_name

        new_env = self.make_venvs(env_cnfg_to_pass)
        self.venv.close()
        self.venv = new_env

        self.observation_space = new_env.observation_space
        self.action_space = new_env.action_space

    def make_venvs(self, cnfg):
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
        return vec_env    
