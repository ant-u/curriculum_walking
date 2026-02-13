import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from scripts.util.skipable_ppo import SkippablePPO


class PerformaneEstimator():
    """Evaluates performance of Agent in currenct env/level.
    This is crucial for CurriculumManager for further training planing.
    """
    
    def __init__(self) -> None:
        pass
    
    def estimate(self, rollout_buffer):
        # raw advantage, unnormalized, see 
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py line 216 - 219, 
        # buffer is not overwritten, normalization uses local copy only
        advantages = rollout_buffer.advantages.copy()
        max_adv = np.maximum(advantages, 0)
        regrets = [max_adv[:,i].sum() / max_adv.shape[0] for i in range(max_adv.shape[1])]
        # print(f"rollout mean regret: {np.mean(regrets)}")
        return regrets
    
    def get_rollout_lenghts(self, buffer):
        episode_lengths = []
        for env_idx in range(buffer.n_envs):
            starts = np.where(buffer.episode_starts[:, env_idx])[0]
            # gaps between starts give completed episode lengths
            lengths = np.diff(starts)
            episode_lengths.extend(lengths.tolist())
            # last incomplete episode (from last start to buffer end)
            if len(starts) > 0:
                episode_lengths.append(buffer.buffer_size - starts[-1])
        return episode_lengths
    
    def collect_scoring_rollout(self, model: SkippablePPO, dummy_callback, n_steps: int):

        scoring_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs
        )

        model.collect_rollouts(
            env=model.env,
            callback=dummy_callback,
            rollout_buffer=scoring_buffer,
            n_rollout_steps=n_steps
        )
        return scoring_buffer