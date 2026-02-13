import os
import numpy as np
import matplotlib.pyplot as plt
from envs.vec_env import make_env, load_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.curriculum_manager import CurriculumManager
from envs.curriculum.level_generator import LevelGenerator
from scripts.util.plot import Plot, IQRPlot
# from envs.vec_env import make_env
# from scripts.train import ENV_CONFIG


class CurriculumCallback(BaseCallback):
    def __init__(self, save_dir: str, env_cnfg: dict, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.env_cnfg = env_cnfg
        self.performance_est = PerformaneEstimator()
        self.level_gen = LevelGenerator(50, [-10,10], 150, 3, 1)
        self.regrets = []
        self.regrent_plot = Plot(title="Regret", xlabel="Steps", 
                            ylabel="Regret", line_label="Avg rollout regret")
        self.rollout_counter = 0

    def _on_training_start(self):
        self.curriculum_manr = CurriculumManager(self.training_env, 100, 1, 0.05, 0.05, 0.05, 42)
        
    def _on_step(self):
        return True
    
    def _on_rollout_start(self):
        self.curriculum_manr.before_rollout()
        
    def _on_rollout_end(self) -> None:
        # self.training_env.env_method("set_env_level_slab", height=1, x_ratio=0.7)  # for calling a method
        regrets = self.performance_est.estimate(self.model)
        mean_regret = np.mean(regrets)
        self.regrets.append(mean_regret)
        self.regrent_plot.update(self.regrets)
        self.regrent_plot.save(os.path.join(self.save_dir, "regret.svg"))

        update_policy = self.curriculum_manr.after_rollout(mean_regret)
        if not update_policy:
            self.model.skip_training = True
        
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
        
        
def get_all_callbacks(callback_cnfg, env_cnfg, run_dir, train_on) -> tuple:
    CHECKPOINT_PATH = os.path.join(run_dir, "checkpoints")
    LOG_PATH = os.path.join(run_dir, "logs")
    save_vec_norm = SaveVecNormalizeOnNewBest(CHECKPOINT_PATH)

    checkpoint_callback = CheckpointCallback(
        save_freq=callback_cnfg["checkpoint_cb_conf"]["save_freq"] // env_cnfg["n_envs"],
        save_path=CHECKPOINT_PATH,
        save_vecnormalize=callback_cnfg["checkpoint_cb_conf"]["save_vecnormalize"],
        name_prefix=callback_cnfg["checkpoint_cb_conf"]["name_prefix"]
    )

    env_cnfg_tmp = env_cnfg.copy()
    env_cnfg_tmp["n_envs"] = callback_cnfg["eval_env_conf"]["n_envs"]
    env_cnfg_tmp["max_steps"] = callback_cnfg["eval_env_conf"]["max_steps"]
    if train_on != None and train_on != "":
        eval_env = load_env(train_on, env_cnfg_tmp)
    else:
        eval_env = make_env(env_cnfg_tmp)
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=callback_cnfg["eval_env_conf"]["n_eval_episodes"],
        best_model_save_path=CHECKPOINT_PATH,
        log_path=LOG_PATH,
        eval_freq=callback_cnfg["eval_env_conf"]["eval_freq"] // env_cnfg["n_envs"],
        deterministic=callback_cnfg["eval_env_conf"]["deterministic"],
        render=callback_cnfg["eval_env_conf"]["render"],
        callback_on_new_best=save_vec_norm,
    )
    plot_callback = LivePlotCallback(
        save_dir=LOG_PATH,
        window=callback_cnfg["plot_callback"]["window"],
        log_level=callback_cnfg["plot_callback"]["log_level"],
    )
    curr_callback = CurriculumCallback(save_dir=LOG_PATH, env_cnfg=env_cnfg)
    return checkpoint_callback, eval_callback, plot_callback, curr_callback


class SaveVecNormalizeOnNewBest(BaseCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        # This is called exactly once per new best
        env = self.model.get_env()
        if hasattr(env, "save"):
            env.save(f"{self.save_path}/best_vecnormalize_stats.pkl")
        return True