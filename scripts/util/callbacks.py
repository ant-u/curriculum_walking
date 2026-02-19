import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from envs.vec_env import make_env, load_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.curriculum_manager import CurriculumManager
from envs.curriculum.level_generator import LevelGenerator
from scripts.util.plot import Plot, IQRPlot, DoubleLinePlot
# from envs.vec_env import make_env
# from scripts.train import ENV_CONFIG


class DummyCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self.reset()

    def reset(self):
        self.total_runs_completed = 0
        self.successfull_runs = 0
        self.all_progress = []

    def get_run_metrics(self):
        return (self.successfull_runs, self.total_runs_completed), self.all_progress

    def _on_step(self):
        if any(self.locals["dones"]):
            done_envs_ids = np.where(self.locals["dones"] == True)[0]
            done_envs = np.array(self.locals["infos"])[done_envs_ids]
            self.all_progress.extend([env["progress"] for env in done_envs])
            self.total_runs_completed += sum(self.locals["dones"])
            if any([x["success"] for x in self.locals["infos"]]):
                self.successfull_runs += sum([x["success"] for x in self.locals["infos"]])
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, save_dir: str, env_cnfg: dict, curr_cnfg: dict, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_buffer_path = os.path.join(self.save_dir, "buffer_logs.txt")
        self.env_cnfg = env_cnfg
        self.curr_cnfg = curr_cnfg
        self.performance_est = PerformaneEstimator()
        self.level_gen = LevelGenerator(50, [-10,10], 150, 3, 1)
        self.regrets = []
        self.regrent_plot = Plot(title="Regret", xlabel="Steps", 
                            ylabel="Regret", line_label="Avg rollout regret")
        self.rollout_counter = 0
        self.eval_rollout_length = curr_cnfg["evaluation_episode_steps"]
        self.dummy_callback = None
        self.successfull_runs = 0
        self.total_runs_completed = 0
        self.all_progress = []

    def _reset_last_obs(self):
        self.model._last_obs = self.model.env.reset()
        self.model._last_episode_starts = np.ones((self.model.n_envs,), dtype=bool)
        self.total_runs_completed = self.successfull_runs = 0
        self.all_progress.clear()
        self.dummy_callback.reset()

    def _on_training_start(self):
        self.curriculum_manr = CurriculumManager(self.training_env, self.curr_cnfg)
        self.dummy_callback = DummyCallback()
        self.dummy_callback.init_callback(self.model)
        
    def _on_step(self):
        if any(self.locals["dones"]):
            done_envs_ids = np.where(self.locals["dones"] == True)[0]
            done_envs = np.array(self.locals["infos"])[done_envs_ids]
            self.all_progress.extend([env["progress"] for env in done_envs])
            self.total_runs_completed += sum(self.locals["dones"])
            if any([x["success"] for x in self.locals["infos"]]):
                self.successfull_runs += sum([x["success"] for x in self.locals["infos"]])
        return True
    
    def _on_rollout_start(self):
        start_training = self.curriculum_manr.before_rollout()
        self._reset_last_obs()
        while not start_training:
            buffer = self.performance_est.collect_scoring_rollout(self.model, self.dummy_callback, self.eval_rollout_length)
            succ_metrics, all_progress = self.dummy_callback.get_run_metrics()
            regrets = self.performance_est.estimate(buffer)
            lengths = self.performance_est.get_rollout_lenghts(buffer)
            self.curriculum_manr.after_rollout(np.mean(regrets), succ_metrics, lengths, all_progress)
            start_training = self.curriculum_manr.before_rollout()
            self._reset_last_obs()

        
    def _on_rollout_end(self) -> None:
        # self.training_env.env_method("set_env_level_slab", height=1, x_ratio=0.7)  # for calling a method
        regrets = self.performance_est.estimate(self.model.rollout_buffer)
        lengths = self.performance_est.get_rollout_lenghts(self.model.rollout_buffer)
        mean_regret = np.mean(regrets)
        self.regrets.append(mean_regret)
        self.regrent_plot.update(self.regrets)
        self.regrent_plot.save(os.path.join(self.save_dir, "regret.svg"))
        self.curriculum_manr.after_rollout(mean_regret, (self.successfull_runs, self.total_runs_completed), lengths, self.all_progress)
        if self.rollout_counter % 5 == 0:
            self.curriculum_manr.dump_buffer_to_file(self.save_buffer_path)
        self.rollout_counter += 1
        
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
        
        
def get_all_callbacks(callback_cnfg, env_cnfg, curr_cnfg, run_dir, train_on) -> tuple:
    CHECKPOINT_PATH = os.path.join(run_dir, "checkpoints")
    LOG_PATH = os.path.join(run_dir, "logs")

    checkpoint_callback = CheckpointCallback(
        save_freq=callback_cnfg["checkpoint_cb_conf"]["save_freq"] // env_cnfg["n_envs"],
        save_path=CHECKPOINT_PATH,
        save_vecnormalize=callback_cnfg["checkpoint_cb_conf"]["save_vecnormalize"],
        name_prefix=callback_cnfg["checkpoint_cb_conf"]["name_prefix"]
    )

    gen = LevelGenerator()
    env_cnfg_tmp = env_cnfg.copy()
    env_cnfg_tmp["n_envs"] = callback_cnfg["eval_env_conf"]["n_envs"]
    env_cnfg_tmp["max_steps"] = callback_cnfg["eval_env_conf"]["max_steps"]
    eval_callbacks = []
    for level in callback_cnfg["eval_env_conf"]["eval_levels"]:
        if train_on != None and train_on != "":
            eval_env = load_env(train_on, env_cnfg_tmp)
        else:
            eval_env = make_env(env_cnfg_tmp)
        eval_env.training = False
        level_elems = gen.create_level_elements(*level["params"], level["seed"])
        level_des = gen.calculate_element_coords(level_elems)
        eval_env.env_method("set_level_template", level=level_des)
        freq = callback_cnfg["eval_env_conf"]["eval_freq"] if level["name"] == "plain" else callback_cnfg["eval_env_conf"]["eval_freq_obstacles"]
        freq = freq // env_cnfg["n_envs"]

        save_vec_norm = SaveVecNormalizeOnNewBest(os.path.join(CHECKPOINT_PATH, level["name"]))
        eval_callback = NamedEvalCallback(
            eval_env,
            n_eval_episodes=callback_cnfg["eval_env_conf"]["n_eval_episodes"],
            best_model_save_path=os.path.join(CHECKPOINT_PATH, level["name"]),
            # log_path=os.path.join(LOG_PATH, level["name"]),
            eval_freq=freq,
            deterministic=callback_cnfg["eval_env_conf"]["deterministic"],
            render=callback_cnfg["eval_env_conf"]["render"],
            callback_on_new_best=save_vec_norm,
            name=level["name"],
            save_path=os.path.join(LOG_PATH, level["name"]),
        )
        eval_callbacks.append(eval_callback)
    plot_callback = LivePlotCallback(
        save_dir=LOG_PATH,
        window=callback_cnfg["plot_callback"]["window"],
        log_level=callback_cnfg["plot_callback"]["log_level"],
    )
    curr_callback = CurriculumCallback(save_dir=LOG_PATH, env_cnfg=env_cnfg, curr_cnfg=curr_cnfg)
    return checkpoint_callback, eval_callbacks, plot_callback, curr_callback


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
    

class NamedEvalCallback(EvalCallback):
    def __init__(self, *args, name: str, save_path: str, **kwargs):
        # TODO: create plot of eval level success rates (or average progress)
        super().__init__(*args, **kwargs)
        self.name = name
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.all_progress = []
        self.total_runs_completed = 0
        self.successfull_runs = 0
        self._env_done = np.zeros(self.eval_env.num_envs, dtype=bool)

        self.accumulated_succ_rate = []
        self.accumulated_total_runs = []
        self.acc_run_number = []
        self.plot = DoubleLinePlot(f"Evaluation of level {self.name}", "time steps", "%", "Average progress",
                                   "Success rate", False)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n------ Evaluating terrain: {self.name} ------")
        to_return = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            succ_r = self.successfull_runs / self.total_runs_completed
            self.accumulated_succ_rate.append(succ_r)
            self.accumulated_total_runs.append(self.all_progress)
            self.acc_run_number.append(self.num_timesteps)
            print(f"Summary terrain {self.name}: success: {self.successfull_runs} / {self.total_runs_completed} ({100*succ_r:.2f} %)")
            print(f"    average progress: {np.mean(self.all_progress)*100:.2f} %")
            print(f"    runs progress: [" + ", ".join([f"{v:.3f}" for v in self.all_progress]) + "]")
            self.all_progress = []
            self.total_runs_completed = 0
            self.successfull_runs = 0
            self._env_done = np.zeros(self.eval_env.num_envs, dtype=bool)
            self._save_eval_results()
        return to_return
        
    def _log_success_callback(self, locals_, globals_):
        to_return = super()._log_success_callback(locals_, globals_)
        dones = locals_["dones"]
        infos = locals_["infos"]
        # TODO: only fastest n episodes are taken. if one takes long and others finished before the second time, long is not counted, 
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and not self._env_done[i]:
                self._env_done[i] = True  # mark as counted
                self.all_progress.append(info.get("progress"))
                self.total_runs_completed += 1
                if info.get("success", False):
                    self.successfull_runs += 1
            elif not done and self._env_done[i]:
                self._env_done[i] = False  # env has reset, ready for next episode
        return to_return
    
    def _save_eval_results(self):
        data = {
            "succ_r": self.accumulated_succ_rate,
            "total_runs": self.accumulated_total_runs,
            "time_steps": self.acc_run_number
        }
        
        path = os.path.join(self.save_path, "eval_results.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        average_prog_total_runs = [np.clip(np.mean(x), 0, 1) for x in self.accumulated_total_runs]
        x_data = np.array(self.acc_run_number) / 1e6
        self.plot.update_with_x(average_prog_total_runs, self.accumulated_succ_rate, x_data, 0)
        self.plot.save(os.path.join(self.save_path, "eval_plot.svg"))