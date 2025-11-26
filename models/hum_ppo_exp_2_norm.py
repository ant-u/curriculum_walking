from stable_baselines3 import PPO
from env.vec_env import make_env
from stable_baselines3.common.vec_env import VecNormalize

def train():
    env = make_env('Humanoid-v5', seed=0, n_envs=8)
    env = VecNormalize(env, norm_reward=True, clip_reward=10, norm_obs=True)
    reward_logger = None
    log_level = 2
    steps = 50e6
    if log_level >= 1:
        weights = env.env.reward_weights
        reward_logger = RewardLoggerCallback(component_names=weights.keys(), 
            total_steps=steps, max_rewards=weights.values(), log_level=log_level)

    model = PPO(
        "MlpPolicy",
        env,
        device='cpu', 
        learning_rate=1e-4,     # ↓ smaller LR
        n_steps=8192,           # ↑ very long rollouts
        batch_size=256,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=1.0,
        gae_lambda=0.90,
        max_grad_norm=0.3,
        n_epochs=15,
        normalize_advantage=True,
        verbose=1
    )
    model.learn(total_timesteps=steps)

    model.save("alternatives/runs/hum_ppo_exp_2_norm/hum_ppo_exp_2_norm")
    env.save("alternatives/runs/hum_ppo_exp_2_norm/env_exp_2_norm")
    env.close()

if __name__ == '__main__':
    train()