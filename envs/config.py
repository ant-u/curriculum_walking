

PPO_CONFIG = {
    "algo": "PPO",
    "policy": "MlpPolicy",
    "device": "cpu",
    "learning_rate": 3e-5,  # was 1e-4
    "n_steps": 8192,
    "batch_size": 512,  # was 256
    "clip_range": 0.1,  # was 0.15
    "ent_coef": 0.03,  # was 0.01
    "vf_coef": 0.5,  # was 1
    "gamma": 0.99,  # default
    "gae_lambda": 0.95,  # was 0.9
    "max_grad_norm": 0.4,  # was 0.3
    "n_epochs": 10,  # was 15
    "normalize_advantage": True,
    "verbose": 1,
    "tensorboard_log": True,
    "timesteps": 300e6,
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
    "stabilize_lidar": True,  # keep point cloud always in same orientation, not adapted to toso orienteation
    "use_levels": True,
    "geom_z_gap": 1e-3,  # default 1e-3 to not match 0 plane and ghost geoms
    "terminate_at_x_border": 60,  # 0 disables it, midd of end platform at 60
    "min_diff_torso_feet": 0.5,  # reset if height diff of feet (averaged) and torso lower than this
    "healthy_z_range": [-10,10],  # range in which torso is allowed to be
    "exclude_absolute_height": True,  # exclude the abs height from observation
    "use_relative_height": False,  # used for lidar, if false, lidar observation is absolute height
    "seed": 0,
    "n_points_x": 6,
    "n_points_y": 5,
    "y_width": 1.5,
    "x_forward": 4,
    "x_start": 0,
    "falling_punishment": -20,  # negative reward given when falling, was -100
    "norm_reward": True,
    "norm_obs": True,
    "clip_reward": 10,  # default: 10
    "env_kwargs": {},
}

CURR_CONFIG = {
    "use_curr": True,
    "buffer_size": 200,
    "buffer_init_fill_ratio": 0.2,
    "buffer_init_lower_cap": [0,0,0,0,0],  # obst, slab, stairs, stump, gap
    "buffer_init_upper_cap": [0.5, 0.1, 0.1, 0.15, 0.15],
    "level_metric": "reg",  # "lrn" (learnability, 1*(1-p), or "reg" based on PVL 
    "mutation_usage": True,  # True for ACCEL-like (with mutation), false for no mutation -> PLR-like
    "mutation_edit_size": [-0.02, 0.03],  # range for one param to be mutated
    "mutation_number": 2,  # number of params that can be mutated at once
    "replay_decision_distribution": [0.1, 0.9],  # [discover, replay], default 0.1 0.9
    "seed": None,  # Seed for CurrManager
    "selection_temp": 0.9,  # temp for selection of levels, high temp uniform sampling, low temp high likelyhood only for high regret levels 
    "evaluation_episode_steps": 2500,  # steps to take for computing regret on a level. measure to safe time when evaluating 
}

CALLBACK_CONFIG = {
    "checkpoint_cb_conf": {
        "save_freq": 3_000_000,
        "save_vecnormalize": True,
        "name_prefix": "ckpt"
    },
    "eval_env_conf": {
        "env_seed": 0,
        "eval_freq": 500_000,            # evaluate on flat env
        "eval_freq_obstacles": 500_000,  # evaluate on all obstacle envs
        "deterministic": True,
        "render": False,
        "max_steps": 2_000,
        "n_envs": 10,    # NOTE: when choosing more than 1, envs are running in parallel. If one env fails twice fast,
                        # Another env having a long (good) run might be not counted
        "n_eval_episodes": 10,  # accounts for n_envs, so 5 n_envs and 5 episodes result in 5 parallel simulations
        "eval_levels": [{
                "name": "plain",
                "params": [0,0,0,0,0],
                "seed": 0
            }
            ,{
                "name": "slabs",
                "params": [1, 0.4, 0, 0, 0],
                "seed": 42
            },{
                "name": "stairs",
                "params": [1, 0, 0.4, 0, 0],
                "seed": 874
            },{
                "name": "stumps",
                "params": [1, 0, 0, 0.4, 0],
                "seed": 21  # irrelevant
            },{
                "name": "gaps",
                "params": [1, 0, 0, 0, 0.5],
                "seed": 21  # irrelevant
            },{
                "name": "easy",
                "params": [0.25, 0.2, 0.2, 0.2, 0.2],
                "seed": 54090
            },{
                "name": "medium",
                "params": [0.5, 0.4, 0.4, 0.4, 0.4],
                "seed": 799
            },{
                "name": "hard",
                "params": [0.75, 0.5, 0.5, 0.5, 0.5],
                "seed": 180
            },{
                "name": "extrem",
                "params": [0.9, 0.8, 0.8, 0.8, 0.8],
                "seed": 52
            }
        ],
    },
    "plot_callback": {
        "window": 100,
        "log_level": 2
    }
}