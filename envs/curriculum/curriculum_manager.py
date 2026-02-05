import numpy as np
from typing import List, Tuple
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.level_generator import LevelDescription, LevelGenerator
import numpy as np
from envs.humanoid_curr import HumanoidEnvCurr


class CurriculumManager:
    """"""
    
    def __init__(self, env, buff_size, buff_ratio, adding_threshold, regret_diff_threshold) -> None:
        """buff size is general buffer size, buff_ratio is inital fill ratio of buffer."""
        self.envs: List[HumanoidEnvCurr] = [e.env for e in env.venv.envs]  # list of wrapped envs
        self.buff_size = buff_size
        self.buff_ratio = buff_ratio
        self.adding_threshold = adding_threshold  # lower threshold for regret-based buffer adding
        self.regret_diff_threshold = regret_diff_threshold  # upper border for regret deviation of a mutation level from parent
        self.buffer: List[LevelDescription] = [None] * self.buff_size
        self.level_gen = LevelGenerator
        self.rng = np.random.default_rng()
        self.replay_decision: bool | None = None
        self.current_level: LevelDescription | None = None
        self.muation_level: bool = False
        self.parent_level_regret: float | None = None
        self._init_buffer()

    def _init_buffer(self):
        for _ in range(self.buff_ratio * self.buff_size):
            n, d = self.sample_level_params()  # cap params for easy init
            self.buffer.append(self.get_level(n, d))

    def before_rollout(self):
        self.replay_decision = bool(self.rng.choice([0,1]))
        if self.muation_level:  # discover mutated replay level
            self.current_level = self._mutate_level(self.current_level)
        elif self.replay_decision:  # learn on buffer level
            buffer_not_none = np.where(self.buffer != None)[0]
            self.current_level = self.rng.choice(buffer_not_none)
        else:  # discover new level
            n, d = self.sample_level_params()
            self.current_level = self.get_level(n, d)
        self._set_level(self.current_level)

    def after_rollout(self, regrets) -> bool:
        """Takes regrets for level, decides wether policy update shall be applied or not."""
        if self.muation_level:
            eval_value = self.parent_level_regret - regrets  # use absolut value, compare with other levels
            if eval_value <= self.regret_diff_threshold:
                self._update_buffer(self.current_level)
            return False
        elif self.replay_decision:  # level from buffer was used
            if regrets >= self.adding_threshold:
                self._update_buffer(self.current_level)
                self.muation_level = True  # Next level is mutation level
                self.parent_level_regret = regrets
            return True
        else:  # new level sample was used
            if regrets >= self.adding_threshold:
                self._update_buffer(self.current_level)
            return False

    def update(self, regrets):
        """Called on every rollout end with regerets from every venv"""
        for i, r in enumerate(regrets):
            level = self.envs[i].current_level  # enum
        # self.env.env_method()

    def get_level(self, n: float, d: float) -> LevelDescription:
        return self.level_gen.create_level(n, d)
    
    def sample_level_params(self) -> Tuple[float]:
        n = self.rng.random()
        d = self.rng.random()
        return n, d
    
    def _set_level(self, level: LevelDescription):
        for e in self.envs:
            e.set_level_template(level)
        self.reset_envs()
        # NOTE: important for accurate GAE values, wihtout reset env, one rollout can contain
        # data from different levels, which is very suboptimal for GAE
    
    def reset_envs(self):
        for e in self.envs:
            e.reset()

    def _update_buffer(self, level: LevelDescription):
        if len(self.buffer) >= self.buff_size:
            self.buffer.pop(0)  # TODO: FIFO strategie, not ideal, ADAPT
        self.buffer.append(level)

    def _mutate_level(self, level: LevelDescription) -> LevelDescription:
        return None
