from dataclasses import dataclass
from typing import List
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.level_generator import LevelType
import numpy as np
from envs.humanoid_curr import HumanoidEnvCurr


class CurriculumManager:
    """Tracks the evaluated skills of the agent for all levels.
    Based on current agent performance, calls LevelGenerator for creating 'next' level
    for the curriculum.  
    """
    
    def __init__(self, env) -> None:
        self.envs: List[HumanoidEnvCurr] = [e.env for e in env.venv.envs]  # list of wrapped envs
        self.performance_est = PerformaneEstimator()
        self.skill_tracker = SkillTracker(list(LevelType))
        self.curr_level = None
        self.threshold = 0.1
        self.change_level = False
        self.level_options = list(LevelType)  # PLANE, SLAB, STAIRS, LOG, STUMP, RAMP
        self.level_probs = [0.1, 0.2, 0.2, 0.1, 0.2, 0.2]

    def update(self, regrets):
        for i, r in enumerate(regrets):
            level = self.envs[i].current_level  # enum
            self.skill_tracker.get_skill(level).update(r)
        # self.env.env_method()

        if self.skill_tracker.get_skill(level).should_increase_difficulty(self.threshold):
            self.change_level = True
        # NOTE: might come more conditions

        if self.change_level:
            next_level = self.get_next_level()
            level_kwargs = self.skill_tracker.get_skill(next_level).get_new_kwargs()
            for e in self.envs:
                e.set_level(next_level, **level_kwargs)
                e.reset()
            # env.set_level
            # env.reset()  
            # NOTE: important for accurate GAE values, wihtout reset env, one rollout can contain
            # data from different levels, which is very suboptimal for GAE

    def get_next_level(self):
        return np.random.choice(self.level_options, p=self.level_probs)
    
    def reset_envs(self):
        for e in self.envs:
            e.reset()


@dataclass
class SkillTracker:

    def __init__(self, levels: list[LevelType]):
        res_len = 2 * len(levels)
        self._skills: dict[LevelType, Skill] = {
            lvl: Skill(lvl.value, result_length=res_len) for lvl in levels}

    def get_skill(self, level: LevelType):
        return self._skills[level]


@dataclass
class Skill:
    level_name: str
    difficulty: float = 0.0
    result_length: int = 10
    prev_results = np.ones(result_length).tolist()
    episodes: int = 0
    regret: float = 0.0

    def update(self, regret: float):
        self.prev_results.pop(0)
        self.prev_results.append(regret)
        self.regret = np.mean(self.prev_results)

    def should_increase_difficulty(self, threshold) -> bool:
        return self.regret >= threshold


    