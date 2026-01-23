import numpy as np

class PerformaneEstimator():
    """Evaluates performance of Agent in currenct env/level.
    This is crucial for CurriculumManager for further training planing.
    """
    
    def __init__(self) -> None:
        pass
    
    def estimate(self, model):
        # raw advantage, unnormalized, see 
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py line 216 - 219, 
        # buffer is not overwritten, normalization uses local copy only
        advantages = model.rollout_buffer.advantages.copy()
        max_adv = np.maximum(advantages, 0)
        regrets = [max_adv[:,i].sum() / max_adv.shape[0] for i in range(max_adv.shape[1])]
        print(f"rollout mean regret: {np.mean(regrets)}")
        return regrets