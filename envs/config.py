

PPO_CONFIG = {
    "algo": "PPO",
    "policy": "MlpPolicy",
    "device": "cpu",
    "learning_rate": 1e-4,
    "n_steps": 4096,
    "batch_size": 256,
    "clip_range": 0.15,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "gamma": 0.99,  # default
    "gae_lambda": 0.90,
    "max_grad_norm": 0.3,
    "n_epochs": 15,
    "normalize_advantage": True,
    "verbose": 1,
    "tensorboard_log": True,
    "timesteps": 200e6,
    "seed": 0,
    "partition": "Krater",  # e.g. NvidiaAll or Krater
}

ENV_CONFIG = {
    "xml_file": "humanoid_plane.xml",  # NOTE: full name of a file in ./models. If empty / none, defaults are used
    "env_id": "HumanoidEnvCurr",  # "HumanoidEnvDefault", "HumanoidEnvBase", "HumanoidEnvCurr"
    "n_envs": 12,
    "max_steps": 0,  # 0 disables max steps
    "use_lidar": True,
    "render_lidar": True,  # NOTE: is disabled during training
    "use_levels": True,
    "geom_z_gap": 1e-3,  # default 1e-3 to not match 0 plane and ghost geoms
    "terminate_at_x_border": 60,  # 0 disables it, midd of end platform at 60
    "min_diff_torso_feet": 0.2,  # reset if height diff of feet (averaged) and torso lower than this
    "use_relative_height": False,
    "seed": 0,
    "n_points_x": 6,
    "n_points_y": 5,
    "y_width": 1.5,
    "x_forward": 4,
    "x_start": 0,
    "norm_reward": True,
    "norm_obs": True,
    "clip_reward": 10,  # default: 10
    "env_kwargs": {},
}

CURR_CONFIG = {
    "buffer_size": 200,
    "buffer_init_fill_ratio": 0.1,
    "buffer_init_lower_cap": [0,0,0,0,0],  # obst, slab, stairs, stump, gap
    "buffer_init_upper_cap": [0.2, 0.1, 0.1, 0.1, 0.1],
    "mutation_edit_size": [-0.02, 0.05],  # range for one param to be mutated
    "mutation_number": 2,  # number of params that can be mutated at once
    "replay_decision_distribution": [0.2, 0.8],  # [discover, replay], default 0.1 0.9
    "seed": None,  # Seed for CurrManager
    "selection_temp": 1.0,  # temp for selection of levels, high temp uniform sampling, low temp high likelyhood only for high regret levels 
    "evaluation_episode_steps": 2000,  # steps to take for computing regret on a level. measure to safe time when evaluating 
}

CALLBACK_CONFIG = {
    "checkpoint_cb_conf": {
        "save_freq": 3_000_000,
        "save_vecnormalize": True,
        "name_prefix": "ckpt"
    },
    "eval_env_conf": {
        "env_seed": 0,
        "eval_freq": 500_000,
        "deterministic": True,
        "render": False,
        "max_steps": 20_000,
        "n_envs": 5,
        "n_eval_episodes": 5,  # accounts for n_envs, so 5 n_envs and 5 episodes result in 5 parallel simulations
    },
    "plot_callback": {
        "window": 100,
        "log_level": 2
    }
}