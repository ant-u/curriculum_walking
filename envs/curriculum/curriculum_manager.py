from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.level_generator import LevelDescription, LevelGenerator, Element, Level
import numpy as np
from envs.humanoid_curr import HumanoidEnvCurr


@dataclass
class BufferLevel():
    """Representing a level based on params and seed, used for buffer. 
    With all params and seed given, level can be identically reconstruced by LevelGenerator.
    Also, level regret is saved here since needed in Buffer."""
    seed: int
    obstacles: float
    diff_slab: float
    diff_stairs: float
    diff_stump: float
    diff_gap: float
    regret: Optional[float] = None
    succ_r: Optional[float] = None

    def sample_attributes(self, n):
        choices = ["obstacles", "diff_slab", "diff_stairs", "diff_stump", "diff_gap"]
        return np.random.choice(choices, n, replace=False)
    
    def copy(self):
        return BufferLevel(seed=self.seed,obstacles=self.obstacles,diff_slab=self.diff_slab,
            diff_stairs=self.diff_stairs,diff_stump=self.diff_stump,diff_gap=self.diff_gap,regret=None)
    
    def __str__(self):
        s = f"BufferLevel with seed={self.seed}, obst={self.obstacles:.5f}, slab={self.diff_slab:.5f}, " +\
            f"stairs={self.diff_stairs:.5f}, stump={self.diff_stump:.5f}, gap={self.diff_gap:.5f}"
        s += f", regeret={self.regret:.5f}" if self.regret else f", regret={self.regret}"
        s += f", succ_r={self.succ_r:.5f}" if self.succ_r else f", succ_r={self.succ_r}"
        return s
    
    def _long_string(self) -> str:
        """returns unshortened string of level for reproducability in logs"""
        return f"BufferLevel with seed={self.seed}, obst={self.obstacles}, slab={self.diff_slab}, " +\
            f"stairs={self.diff_stairs}, stump={self.diff_stump}, gap={self.diff_gap} regret={self.regret} succ_r={self.succ_r}"


class CurriculumManager:
    """"""
    
    def __init__(self, env, cnfg) -> None:
        """buff size is general buffer size, buff_ratio is inital fill ratio of buffer."""
        self.envs: List[HumanoidEnvCurr] = [e.env for e in env.venv.envs]  # list of wrapped envs
        self.buff_size = cnfg["buffer_size"]
        self.buff_ratio = cnfg["buffer_init_fill_ratio"]
        self.buffer_init_lower_cap = cnfg["buffer_init_lower_cap"]
        self.buffer_init_upper_cap = cnfg["buffer_init_upper_cap"]
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

    @property
    def threshold(self) -> float:
        valid = [lvl.regret for lvl in self._get_nonempty_buffer() if lvl.regret is not None]
        if not valid:
            return 0  # minimal regret (per definition positive, lowest possible regret -> 0)
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

    def after_rollout(self, regret, lengths=None, all_runs: Tuple=None, all_progress: Tuple=None):
        """Takes regrets for level, decides wether policy update shall be applied or not."""
        self.current_level.regret = regret
        self.current_level.succ_r = all_runs[0]/all_runs[1] if all_runs else None

        if self.muation_level:
            added = self._try_update_buffer(self.current_level)
            acceptance_str = "ACCEPTED" if added else "DENIED"
            self.muation_level = False
        elif self.replay_decision:  # level from buffer was used
            acceptance_str = "DONE"
            self.muation_level = True  # Next level is mutation level
        else:  # new level sample was used
            added = self._try_update_buffer(self.current_level)
            acceptance_str = "ACCEPTED" if added else "DENIED"
        self._print_rollout_summary(regret, lengths, all_runs, all_progress, acceptance_str, self.threshold)

    def sample_from_buffer(self, temp=1.0) -> BufferLevel:
        """pick random sample from buffer, weighted with regret values. 
        High temp -> uniform chance, low temp -> only high regrets very likely."""
        valid = self._get_nonempty_buffer()
        valid.sort(key=lambda x: (x.regret is not None, x.regret))
        if not valid:
            return None
        regrets = np.array([lvl.regret if lvl.regret is not None else 0.0 for lvl in valid])
        shifted = regrets - regrets.max()
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
        """update buffer if level is good enough. Regret must be contained in level object.
        Returns True if added and False otherwise."""
        good_enoug = level.regret > self.threshold
        if good_enoug:
            self._update_buffer(level)
        return good_enoug
    
    def _update_buffer(self, level):
        """update buffer with given level, make sure buffer length is staying same. Sorting out lowest regret member when overflow."""
        if len(self.buffer) >= self.buff_size:
            if None in self.buffer:
                self.buffer.remove(None)
            else:
                self.buffer.sort(key=lambda x: (x.regret is not None, x.regret))
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
    
    def dump_buffer_to_file(self, path):
        reduced_buffer = list(filter(None, self.buffer))
        n_nones = sum(x is None for x in self.buffer)
        with open(path, "a") as f:
            f.write(f"None * {n_nones}\n")
            for level in reduced_buffer:
                f.write(level._long_string() + "\n")
            f.write("\n---------------------------------------\n\n")

    def _print_rollout_summary(self, regret, lengths, all_runs, all_progress, acceptance_str, print_threshold):
        to_print = f"    {acceptance_str}:  regret score was {regret:.5f}, threshold at {print_threshold:.5f}."
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
