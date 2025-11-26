import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt

class RewardLoggerCallback(BaseCallback):
    def __init__(self, component_names, total_steps, max_rewards=None,
                 window=100, update_freq=1000, verbose=0, log_level=2):
        super().__init__(verbose)
        self.component_names = component_names
        self.max_rewards = max_rewards or [None] * len(component_names)
        self.window = window
        self.update_freq = update_freq
        self.log_level = log_level

        # Buffers for per-step component rewards
        self.reward_histories = [[] for _ in component_names]
        self.mean_histories = [{'avg': [], 'q25': [], 'q75': []} for _ in component_names]
        self.steps = []

        # Buffers for per-episode stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.avg_episode_rewards = {'avg': [], 'q25': [], 'q75': []}
        self.avg_episode_lengths = []
        self.fill = None
        self.fill_multi = [None for _ in component_names]

        # === Figure 1: Reward components ===
        plt.ion()
        self.fig1, self.axes1 = plt.subplots(
            len(component_names), 1, figsize=(8, 3 * len(component_names))
        )
        if len(component_names) == 1:
            self.axes1 = [self.axes1]

        self.lines1 = []
        for ax, name, max_r in zip(self.axes1, component_names, self.max_rewards):
            line, = ax.plot([], [], label=f"{name} mean", color="tab:blue")
            if max_r is not None:
                ax.axhline(max_r, color="red", linestyle="--", label="max")
            ax.set_title(name)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Avg Reward")
            ax.legend()
            self.lines1.append(line)
        self.fig1.tight_layout()

        # === Figure 2: Episode averages ===
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5))
        self.line_length, = self.ax2.plot([], [], label="Avg episode length", color="tab:blue")
        self.line_reward, = self.ax2.plot([], [], label="Avg total reward", color="tab:red")
        self.ax2.set_title(f"Episode Statistics ({total_steps} steps)")
        self.ax2.set_xlabel("Steps")
        self.ax2.set_ylabel("Value")
        self.ax2.legend()
        self.fig2.tight_layout()

    def _on_step(self):
        info = self.locals["infos"][0]

        # === Step-level reward components ===
        if "reward_components" in info:
            components = np.array(list(info["reward_components"].values()), dtype=float)
            for i, val in enumerate(components):
                self.reward_histories[i].append(val)
                if "episode" in info:  # if len(self.reward_histories[i]) >= self.window:
                    mean_val = np.mean(self.reward_histories[i])
                    self.mean_histories[i]['avg'].append(mean_val)
                    self.mean_histories[i]['q25'].append(np.percentile(self.reward_histories[i], 25))
                    self.mean_histories[i]['q75'].append(np.percentile(self.reward_histories[i], 75))
                    self.reward_histories[i] = []

        # === Episode-level stats ===
        
        if "episode" in info:
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

        # === Update plots occasionally ===
        if self.log_level >= 2:
            if self.num_timesteps % self.update_freq == 0 and self.num_timesteps > 0:
                self._update_plots()
        return True

    def _update_plots(self):
        # --- Reward component plots ---
        for i, (line, ax) in enumerate(zip(self.lines1, self.axes1)):
            x = np.arange(len(self.mean_histories[i]['avg']))
            y_avg = self.mean_histories[i]['avg']
            y_q25 = self.mean_histories[i]['q25']
            y_q75 = self.mean_histories[i]['q75']
            line.set_data(x, y_avg)
            if self.fill_multi[i] is not None:
                self.fill_multi[i].remove()
            self.fill_multi[i] = ax.fill_between(x, y_q25, y_q75, color="blue", alpha=0.2, label="IQR")
            ax.relim()
            ax.autoscale_view()
            ymin, ymax = ax.get_ylim()
            ymin = min(ymin, 0)  # pick ymin if its smaller than 0
            ax.set_ylim(bottom=ymin, top=ymax)

        # --- Episode averages plot ---
        x2 = np.arange(len(self.avg_episode_rewards['avg']))
        self.line_reward.set_data(x2, self.avg_episode_rewards['avg'])
        y_q25 = self.avg_episode_rewards['q25']
        y_q75 = self.avg_episode_rewards['q75']
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.ax2.fill_between(x2, y_q25, y_q75, color="red", alpha=0.2, label="IQR")
        self.line_length.set_data(x2, self.avg_episode_lengths)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Draw all
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
    
    def save_plot(self, path: str):
        self._update_plots()
        self.fig1.savefig(path + "_reward_comp.svg")
        self.fig2.savefig(path + "_total_reward.svg")