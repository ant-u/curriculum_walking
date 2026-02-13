import os
from scripts.util.skipable_ppo import SkippablePPO

def get_PPO(cnfg, env, run_dir):
    LOG_PATH = os.path.join(run_dir, "logs")
    model = SkippablePPO(
        policy              = cnfg["policy"],
        env                 = env,
        device              = cnfg["device"],
        learning_rate       = cnfg["learning_rate"],
        n_steps             = cnfg["n_steps"],
        batch_size          = cnfg["batch_size"],
        clip_range          = cnfg["clip_range"],
        ent_coef            = cnfg["ent_coef"],
        vf_coef             = cnfg["vf_coef"],
        gamma               = cnfg["gamma"],
        gae_lambda          = cnfg["gae_lambda"],
        max_grad_norm       = cnfg["max_grad_norm"],
        n_epochs            = cnfg["n_epochs"],
        normalize_advantage = cnfg["normalize_advantage"],
        verbose             = cnfg["verbose"],
        tensorboard_log     = (LOG_PATH if cnfg["tensorboard_log"] else None),
        seed                = cnfg["seed"]
    )
    return model


def load_PPO(cnfg, env, ppo_dir, run_dir):
    """Loading PPO from the 'last_model' found under checkpoints of given ppo_dir location."""
    LOG_PATH = os.path.join(run_dir, "logs")
    ppo_path = os.path.join(ppo_dir, 'checkpoints', 'last_model.zip')
    model = SkippablePPO.load(
        path=ppo_path,
        env=env,
        device=cnfg["device"],
        tensorboard_log= (LOG_PATH if cnfg["tensorboard_log"] else None)
    )
    print(f"using pretrained model from: {ppo_dir}")
    return model