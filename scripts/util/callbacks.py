import os
import numpy as np
import matplotlib.pyplot as plt
from envs.vec_env import make_env_base, make_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.curriculum_manager import CurriculumManager
from envs.curriculum.level_generator import LevelGenerator
from scripts.util.plot import Plot, IQRPlot


class CurriculumCallback(BaseCallback):
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.performance_est = PerformaneEstimator()
        self.curriculum_manr = CurriculumManager()
        self.level_gen = LevelGenerator()
        self.regrets = []
        self.regrent_plot = Plot(title="Regret", xlabel="Steps", 
                            ylabel="Regret", line_label="Avg rollout regret")
        
    def _on_step(self):
        return True
        
    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        
        # raw advantage, unnormalized, see 
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py line 216 - 219, 
        # buffer is not overwritten, normalization uses local copy only
        advantages = buffer.advantages.copy()
        regret = np.maximum(advantages, 0).sum() / advantages.shape[0]
        self.regrets.append(regret)
        self.regrent_plot.update(self.regrets)
        self.regrent_plot.save(os.path.join(self.save_dir, "regret.svg"))
        print(f"rollout mean regret: {self.regrets[-1]}")
        # self.training_env.env_method("set_env_level_slab", height=1, x_ratio=0.7)  # for calling a method
        self.performance_est.estimate(advantages)
        
    def _on_training_end(self):
        self.regrent_plot.update(self.regrets)
        self.regrent_plot.save(os.path.join(self.save_dir, "regret.svg"))
        

class LivePlotCallback(BaseCallback):
    def __init__(self, save_dir: str, window=100, update_freq=20000, verbose=0, log_level=2):
        super().__init__(verbose)
        self.save_dir = save_dir  # folder where plot(s) shall be saved to
        self.window = window  # taking average off all {window} episodes
        self.update_freq = update_freq
        self.log_level = log_level

        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_episode_rewards = {'avg': [], 'q25': [], 'q75': []}
        self.avg_episode_lengths = []
        self.fill = None  # saving the q25 to q75 shadings here
        
        # Plot for episode length and total reward
        plt.ion()
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5))
        self.line_length, = self.ax2.plot([], [], label="Avg episode length", color="tab:blue")
        self.line_reward, = self.ax2.plot([], [], label="Avg total reward", color="tab:red")
        self.ax2.set_title(f"Episode Statistics")
        self.ax2.set_xlabel("Episodes")
        self.ax2.set_ylabel("Value")
        self.ax2.legend()
        self.fig2.tight_layout()
        
    def _on_step(self) -> bool:
        # Monitor returns episode info only when a full episode ends
        for info in self.locals["infos"]:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
                if len(self.episode_rewards) >= self.window:
                    mean_r = np.mean(self.episode_rewards)
                    self.avg_episode_rewards['avg'].append(mean_r)
                    self.avg_episode_rewards['q25'].append(np.percentile(self.episode_rewards, 25))
                    self.avg_episode_rewards['q75'].append(np.percentile(self.episode_rewards, 75))
                    self.episode_rewards = []


                self.episode_lengths.append(info["episode"]["l"])
                if len(self.episode_lengths) >= self.window:
                    mean_l = np.mean(self.episode_lengths[-self.window:])
                    self.avg_episode_lengths.append(mean_l)
                    self.episode_lengths = []
        
        # refresh plot
        if self.log_level >= 2:
            if self.num_timesteps % self.update_freq == 0 and self.num_timesteps > 0:
                self._plot()
        return True

    def _plot(self):
        # --- Episode averages plot ---
        x2 = np.arange(len(self.avg_episode_rewards['avg'])) * self.window  # converting to episodes
        self.line_reward.set_data(x2, self.avg_episode_rewards['avg'])
        y_q25 = self.avg_episode_rewards['q25']
        y_q75 = self.avg_episode_rewards['q75']
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.ax2.fill_between(x2, y_q25, y_q75, color="red", alpha=0.2, label="IQR")
        self.line_length.set_data(x2, self.avg_episode_lengths)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        self.save_plot()
        
    def _on_training_end(self):
        self._plot()
        self.save_plot()
        
    def save_plot(self):
        self.fig2.savefig(os.path.join(self.save_dir, "episode_len_reward.svg"))
        
        
def get_all_callbacks(callback_cnfg, env_cnfg, run_dir) -> tuple:
    CHECKPOINT_PATH = os.path.join(run_dir, "checkpoints")
    LOG_PATH = os.path.join(run_dir, "logs")

    checkpoint_callback = CheckpointCallback(
        save_freq=callback_cnfg["checkpoint_cb_conf"]["save_freq"] // env_cnfg["n_envs"],
        save_path=CHECKPOINT_PATH,
        save_vecnormalize=callback_cnfg["checkpoint_cb_conf"]["save_vecnormalize"],
        name_prefix=callback_cnfg["checkpoint_cb_conf"]["name_prefix"]
    )

    eval_env = make_env(env_cnfg)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_PATH,
        log_path=LOG_PATH,
        eval_freq=callback_cnfg["eval_env_conf"]["eval_freq"] // env_cnfg["n_envs"],
        deterministic=callback_cnfg["eval_env_conf"]["deterministic"],
        render=callback_cnfg["eval_env_conf"]["render"],
    )
    plot_callback = LivePlotCallback(
        save_dir=LOG_PATH,
        window=callback_cnfg["plot_callback"]["window"],
        log_level=callback_cnfg["plot_callback"]["log_level"],
    )
    curr_callback = CurriculumCallback(save_dir=LOG_PATH)
    return checkpoint_callback, eval_callback, plot_callback, curr_callback