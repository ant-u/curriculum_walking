from dataclasses import dataclass
import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.level_generator import LevelDescription, LevelGenerator, Element, Level
import numpy as np
from envs.humanoid_curr import HumanoidEnvCurr
from scripts.util.plot import FiveLinePlot


@dataclass
class BufferLevel():
    """Representing a level based on params and seed, used for buffer. 
    With all params and seed given, level can be identically reconstruced by LevelGenerator.
    Also, level regret and succ_rate are saved here since needed in Buffer."""
    seed: int
    obstacles: float
    diff_slab: float
    diff_stairs: float
    diff_stump: float
    diff_gap: float
    regret: Optional[float] = None
    succ_r: Optional[float] = None
    learnability: Optional[float] = None

    def update_learnability(self, succ_r):
        """update objects succ_r and learnability based on succ_r"""
        self.succ_r = succ_r
        self.learnability = self.succ_r * (1 - self.succ_r)

    def metric(self, metric: str) -> Optional[float]:
        """either 'reg' or 'lrn' for regret or succes-rate based learnability. Possibly None"""
        if metric == "reg":
            return self.regret
        elif metric == "lrn":
            return self.learnability

    def sample_attributes(self, n):
        """Get any of the attributes from the following: obstacles, slab, stairs, stump, gap."""
        choices = ["obstacles", "diff_slab", "diff_stairs", "diff_stump", "diff_gap"]
        return np.random.choice(choices, n, replace=False)
    
    def get_general_difficulty(self) -> float:
        """Get general difficulty as n_obst * sum(difficulties), normed to [0..1]"""
        sum_diff = self.diff_slab + self.diff_stairs + self.diff_stump + self.diff_gap
        return self.obstacles * sum_diff / 4

    def copy(self):
        return BufferLevel(seed=self.seed,obstacles=self.obstacles,diff_slab=self.diff_slab,
            diff_stairs=self.diff_stairs,diff_stump=self.diff_stump,diff_gap=self.diff_gap,regret=None,succ_r=None)

    def __str__(self):
        s = f"BufferLevel with seed={self.seed}, obst={self.obstacles:.5f}, slab={self.diff_slab:.5f}, " +\
            f"stairs={self.diff_stairs:.5f}, stump={self.diff_stump:.5f}, gap={self.diff_gap:.5f}"
        s += f", regeret={self.regret:.5f}" if self.regret else f", regret={self.regret}"
        s += f", learnability={self.learnability:.5f}" if self.learnability else f", learnability={self.learnability}"
        s += f", succ_r={self.succ_r:.5f}" if self.succ_r else f", succ_r={self.succ_r}"
        return s

    def _long_string(self) -> str:
        """returns unshortened string of level for reproducability in logs"""
        return f"BufferLevel with seed={self.seed}, obst={self.obstacles}, slab={self.diff_slab}, stairs={self.diff_stairs}, " +\
            f"stump={self.diff_stump}, gap={self.diff_gap}, regret={self.regret}, learnability={self.learnability}, succ_r={self.succ_r}"


class CurriculumManager:
    """"""

    def __init__(self, env, cnfg) -> None:
        """buff size is general buffer size, buff_ratio is inital fill ratio of buffer."""
        self.envs: List[HumanoidEnvCurr] = [e.env for e in env.venv.envs]  # list of wrapped envs
        self.buff_size = cnfg["buffer_size"]
        self.buff_ratio = cnfg["buffer_init_fill_ratio"]
        self.buffer_init_lower_cap = cnfg["buffer_init_lower_cap"]
        self.buffer_init_upper_cap = cnfg["buffer_init_upper_cap"]
        self.level_metric = cnfg["level_metric"]
        assert self.level_metric in ["reg", "lrn"], "metric must be 'reg' or 'lrn'"
        self.mutation_usage = cnfg["mutation_usage"]
        self.mutation_edit_range = cnfg["mutation_edit_size"]
        self.mutation_number = cnfg["mutation_number"]
        self.temp = cnfg["selection_temp"]
        self.replay_dec_distrib = cnfg["replay_decision_distribution"]

        self.buffer: List[Optional[BufferLevel]] = [None] * self.buff_size
        self.level_gen = LevelGenerator()
        self.rng = np.random.default_rng(cnfg["seed"])
        self.replay_decision: Optional[bool] = None
        self.current_level: Optional[BufferLevel] = None
        self.muation_level: bool = False
        self._init_buffer()
        self.difficulty_thresholds = [0.1, 0.2, 0.4, 0.6, 1]
        self.difficulty_logs = []
        line_lables = [f"Diff under {x}" for x in self.difficulty_thresholds]
        self.diff_ratio_plot = FiveLinePlot("Difficulty ratios", "Buffer updates", "Ratio in %", line_lables, False)

    @property
    def threshold(self) -> float:
        valid = [lvl.metric(self.level_metric) for lvl in self._get_nonempty_buffer() if lvl.metric(self.level_metric) is not None]
        if not valid:
            return 0  # minimal regret (per definition positive, lowest possible regret -> 0) or min learnability
        return min(valid)

    def _init_buffer(self):
        for _ in range(int(self.buff_ratio * self.buff_size)):
            # cap params for easy init
            buffer_level = self.generate_level(self.buffer_init_lower_cap, self.buffer_init_upper_cap)  
            self._update_buffer(buffer_level)

    def before_rollout(self) -> bool:
        """Decides whether training may start or only a value estimation rollout is done. True indicates training can start."""
        # TODO: sometimes on default level to prevent catastrophic forgetting?
        # TODO: is it good idea to always pick levels which have no regret over ones with regret? faster start maybe? or only evaluate all levels in beginning?
        self.replay_decision = bool(self.rng.choice([0,1], p=self.replay_dec_distrib))
        if self.muation_level:  # discover mutated replay level
            self.current_level = self._mutate_level(self.current_level)
            start_training = False
            mode = "MUTATION"
        elif self.replay_decision:  # learn on buffer level
            self.current_level = self.sample_from_buffer(self.temp)
            start_training = True
            mode = "REPLAY - TRAINING"
        else:  # discover new level
            self.current_level = self.generate_level()
            start_training = False
            mode = "DISCOVER"

        self._set_level(self.current_level)
        print(f"{mode} using level: {self.current_level}")
        return start_training

    def after_rollout(self, regret, all_runs: Tuple, lengths=None, all_progress: Tuple=None):
        """Takes metric for level, decides wether policy update shall be applied or not."""
        self.current_level.regret = regret
        self.current_level.update_learnability(all_runs[0]/all_runs[1])

        if self.muation_level:
            added = self._try_update_buffer(self.current_level)
            acceptance_str = "ACCEPTED" if added else "DENIED"
            self.muation_level = False
        elif self.replay_decision:  # level from buffer was used
            acceptance_str = "DONE"
            self.muation_level = True if self.mutation_usage else False  # Next level is mutation level
        else:  # new level sample was used
            added = self._try_update_buffer(self.current_level)
            acceptance_str = "ACCEPTED" if added else "DENIED"
        self._print_rollout_summary(self.current_level.metric(self.level_metric), lengths, all_runs, all_progress, acceptance_str, self.threshold)
        if added:
            self.difficulty_logs.append(self.calc_buffer_difficulties())
            self.update_plot()

    def sample_from_buffer(self, temp=1.0) -> BufferLevel:
        """pick random sample from buffer, weighted with metric values. 
        High temp -> uniform chance, low temp -> only high metric very likely."""
        valid = self._get_nonempty_buffer()
        valid.sort(key=lambda x: (x.metric(self.level_metric) is not None, x.metric(self.level_metric)))
        if not valid:
            return None
        metrics = np.array([lvl.metric(self.level_metric) if lvl.metric(self.level_metric) is not None else 0.0 for lvl in valid])
        shifted = metrics - metrics.max()
        weights = np.exp(shifted / temp)
        weights /= weights.sum()
        sample = self.rng.choice(valid, p=weights)
        return sample

    def generate_level(self, min_params=[0,0,0,0,0], max_params=[1,1,1,1,1], seed=None) -> BufferLevel:
        """Sample level params in order: obstacles, slab, stairs, stump, gap. 
        If seed is not given, it is assigned randomly. If given, it is saved in buffer level."""
        if not seed:
            seed = self.rng.integers(np.iinfo(np.int64).max)  # ~[0, max_int_64]
        params = []
        for min, max in zip(min_params, max_params):
            params.append(self.rng.uniform(min, max))
        b_level = BufferLevel(seed, params[0], params[1], params[2], params[3], params[4])
        return b_level
    
    def _buffer_level_to_level(self, bl: BufferLevel) -> Level:
        """Convert buffer level into Level (description of elemens), based on seed. """
        return self.level_gen.create_level_elements(bl.obstacles, bl.diff_slab, bl.diff_stairs, bl.diff_stump, bl.diff_gap, bl.seed)
    
    def _set_level(self, buffer_level: BufferLevel):
        """Set a buffer level in the envs, therefore converts to Level and then LevelDescription, uses env set_level_template."""
        level = self._buffer_level_to_level(buffer_level)
        level_des = self.level_gen.calculate_element_coords(level)
        for e in self.envs:
            e.set_level_template(level_des)
        self.reset_envs()
        # NOTE: important for accurate GAE values, wihtout reset env, one rollout can contain
        # data from different levels, which is very suboptimal for GAE
    
    def reset_envs(self):
        """Resets all venvs"""
        for e in self.envs:
            e.reset()

    def _try_update_buffer(self, level: BufferLevel) -> bool:
        """update buffer if level is good enough. metric must be contained in level object.
        Returns True if added and False otherwise."""
        good_enoug = level.metric(self.level_metric) > self.threshold
        if good_enoug:
            self._update_buffer(level)
        return good_enoug
    
    def _update_buffer(self, level):
        """update buffer with given level, make sure buffer length is staying same. Sorting out lowest metric member when overflow."""
        if len(self.buffer) >= self.buff_size:
            if None in self.buffer:
                self.buffer.remove(None)
            else:
                self.buffer.sort(key=lambda x: (x.metric(self.level_metric) is not None, x.metric(self.level_metric)))
                self.buffer.pop(0)
        self.buffer.append(level)

    def _get_nonempty_buffer(self) -> List[BufferLevel]:
        """Sort out all None elements"""
        return list(filter(None, self.buffer))

    def _mutate_level(self, buffer_level: BufferLevel) -> BufferLevel:
        """Randomly pick some parameters (obstacles, difficulites) and mutate them by given range"""
        mutation = buffer_level.copy()
        param_str = buffer_level.sample_attributes(self.mutation_number)
        for elem in param_str:
            param = getattr(buffer_level, elem)
            adaption = self.rng.uniform(self.mutation_edit_range[0], self.mutation_edit_range[1])
            updated_param = np.clip(param+adaption, 0, 1)  # making sure [0..1] is never left
            setattr(mutation, elem, updated_param)
        return mutation
    
    def calc_buffer_difficulties(self):
        """return buffer ratio in order of self.difficulty_thresholds. Ratio is calculated from not None levels."""
        difficulties = []
        ratios = []
        for lvl in self._get_nonempty_buffer():
            difficulties.append(lvl.get_general_difficulty())
        lower_diff = 0
        difficulties = np.array(difficulties)
        for threshold in self.difficulty_thresholds:
            ratio = len(np.where((difficulties < threshold) & (difficulties >= lower_diff))[0]) / len(difficulties)
            ratios.append(ratio)
            lower_diff = threshold
        ratios[-1] += len(np.where(difficulties == threshold)[0]) / len(difficulties)  # add elements exactly on one manually
        return ratios
        
    
    def dump_buffer_to_file(self, path):
        #  TODO: log buffer zusammensetzung in discrete categories (easy medium hard extreme oä)
        reduced_buffer = list(filter(None, self.buffer))
        reduced_buffer.sort(key=lambda x: (x.metric(self.level_metric) is not None, x.metric(self.level_metric)))
        n_nones = sum(x is None for x in self.buffer)
        with open(path, "a") as f:
            f.write(f"None * {n_nones}\n")
            for level in reduced_buffer:
                f.write(level._long_string() + "\n")
            f.write("\n---------------------------------------\n\n")

    def save_to_file(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_infos(self, base_path):
        """Save buffer snapshort and history of difficulty ratios to pkl file."""
        self.save_to_file(os.path.join(base_path, "buffer_snapshot.pkl"), self.buffer)
        self.save_to_file(os.path.join(base_path, "difficulty_ratios.pkl"), self.difficulty_logs)
        self.diff_ratio_plot.save(os.path.join(base_path, "difficulty_ratios.svg"))

    def update_plot(self):
        data = []
        for i in range(len(self.difficulty_thresholds)):
            data.append(np.array(self.difficulty_logs)[:, i])
        self.diff_ratio_plot.update(*data, y_lim=0)

    def _print_rollout_summary(self, metric, lengths, all_runs, all_progress, acceptance_str, print_threshold):
        to_print = f"    {acceptance_str}:  {self.level_metric} score was {metric:.5f}, threshold at {print_threshold:.5f}."
        if lengths:
            lengths.sort()
            to_print2 = f"        Avg. rollout length at {np.mean(lengths):.3f}, top 3 runs {lengths[-3:]}"
        if all_runs:
            to_print2 += f" - Successfull runs: {all_runs[0]} / {all_runs[1]} ({(100*all_runs[0]/all_runs[1]):.3f} %)"
        print(to_print)
        print(to_print2)
        if all_progress:
            prog = np.array(all_progress)
            to_print = f"        Avg. run progress: {np.mean(prog):.4f}, of which {(prog >= 0.25).mean():.4f} >= 0.25, {(prog >= 0.5).mean():.4f} >= 0.5, "
            to_print += f" {(prog >= 0.75).mean():.4f} >= 0.75, {(prog >= 0.92).mean():.4f} >= 0.92, {(prog >= 1).mean():.4f} >= 1"
            all_progress.sort()
            to_print2 = f"        Top runs: [" + ", ".join([f"{v:.3f}" for v in all_progress[-5:]]) + "]"
            print(to_print)
            print(to_print2)
