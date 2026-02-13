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

    def sample_attribute(self): 
        choices = ["obstacles", "diff_slab", "diff_stairs", "diff_stump", "diff_gap"]
        return np.random.choice(choices)
    
    def copy(self):
        return BufferLevel(seed=self.seed,obstacles=self.obstacles,diff_slab=self.diff_slab,
            diff_stairs=self.diff_stairs,diff_stump=self.diff_stump,diff_gap=self.diff_gap,regret=self.regret)
    
    def __str__(self):
        s = f"BufferLevel with seed={self.seed}, obst={self.obstacles:.5f}, slab={self.diff_slab:.5f}, " +\
            f"stairs={self.diff_stairs:.5f}, stump={self.diff_stump:.5f}, gap={self.diff_stump:.5f}"
        if self.regret:
            s += f", regeret={self.regret:.5f}"
        else:
            s += f", regret={self.regret}"
        return s



class CurriculumManager:
    """"""
    
    def __init__(self, env, cnfg) -> None:
        """buff size is general buffer size, buff_ratio is inital fill ratio of buffer."""
        self.envs: List[HumanoidEnvCurr] = [e.env for e in env.venv.envs]  # list of wrapped envs
        self.buff_size = cnfg["buffer_size"]
        self.buff_ratio = cnfg["buffer_init_fill_ratio"]
        self.buffer_init_lower_cap = cnfg["buffer_init_lower_cap"]
        self.buffer_init_upper_cap = cnfg["buffer_init_upper_cap"]
        self.mutation_edit_size = cnfg["mutation_edit_size"]
        self.adding_threshold = cnfg["regret_threshold_buffer"]  # lower threshold for regret-based buffer adding
        self.replay_dec_distrib = cnfg["replay_decision_distribution"]

        self.buffer: List[Optional[BufferLevel]] = [None] * self.buff_size
        self.level_gen = LevelGenerator()
        self.rng = np.random.default_rng(cnfg["seed"])
        self.replay_decision: Optional[bool] = None
        self.current_level: Optional[BufferLevel] = None
        self.muation_level: bool = False
        self.parent_level_regret: Optional[float] = None
        self._init_buffer()

    def _init_buffer(self):
        for _ in range(int(self.buff_ratio * self.buff_size)):
            # cap params for easy init
            buffer_level = self.sample_level(self.buffer_init_lower_cap, self.buffer_init_upper_cap)  
            self._update_buffer(buffer_level)

    def before_rollout(self):
        self.replay_decision = bool(self.rng.choice([0,1], p=self.replay_dec_distrib))
        if self.muation_level:  # discover mutated replay level
            self.current_level = self._mutate_level(self.current_level)
        elif self.replay_decision:  # learn on buffer level
            buffer_not_none = list(filter(None, self.buffer))
            self.current_level = self.rng.choice(buffer_not_none)
        else:  # discover new level
            self.current_level = self.sample_level()
        self._set_level(self.current_level)
        print(f"using level: {self.current_level}")

    def after_rollout(self, regret) -> bool:
        """Takes regrets for level, decides wether policy update shall be applied or not."""
        self.current_level.regret = regret
        print(f"after rollout level: {self.current_level}")
        if self.muation_level:
            if regret >= self.adding_threshold:
                self._update_buffer(self.current_level)
            return False
        elif self.replay_decision:  # level from buffer was used
            if regret >= self.adding_threshold:
                self.muation_level = True  # Next level is mutation level
                self.parent_level_regret = regret
            return True
        else:  # new level sample was used
            if regret >= self.adding_threshold:
                self._update_buffer(self.current_level)
            return False

    def sample_level(self, min_params=[0,0,0,0,0], max_params=[1,1,1,1,1], seed=None) -> BufferLevel:
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

    def _update_buffer(self, level: Level):
        if len(self.buffer) >= self.buff_size:
            if None in self.buffer:
                self.buffer.remove(None)
            else:
                self.buffer.sort(key=lambda x: (x.regret is not None, x.regret))
                self.buffer.pop(0)
        self.buffer.append(level)

    def _mutate_level(self, buffer_level: BufferLevel) -> BufferLevel:
        """Randomly pick some parameters (obstacles, difficulites) and mutate them by given range"""
        mutation = buffer_level.copy()
        param_str = buffer_level.sample_attribute()
        param = getattr(buffer_level, param_str)
        adaption = self.rng.uniform(-self.mutation_edit_size, self.mutation_edit_size)
        updated_param = np.clip(param+adaption, 0, 1)  # making sure [0..1] is never left
        setattr(mutation, param_str, updated_param)
        return mutation
