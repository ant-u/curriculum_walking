from dataclasses import dataclass
from envs.curriculum.performance_estimator import PerformaneEstimator
from envs.curriculum.level_generator import LevelType
import numpy as np



class CurriculumManager:
    """Tracks the evaluated skills of the agent for all levels.
    Based on current agent performance, calls LevelGenerator for creating 'next' level
    for the curriculum.  
    """
    
    def __init__(self, env) -> None:
        self.env = env
        self.performance_est = PerformaneEstimator()
        self.skill_tracker = SkillTracker(list(LevelType))
        self.curr_level = LevelType.PLANE  # Default plane to start with eval of walking

    def update(self, regrets):
        for r in regrets:
            pass
            # TODO: add regrets to env individually to keep varying level envs consistent with skills
        self.skill_tracker.get_skill(self.curr_level).update(regret)
        print(self.env.last_level)
        self.env.env_method()


        if regret <= self.regert_threshold:
            self.set_level
            env.set_level
            env.reset()  
            # NOTE: important for accurate GAE values, wihtout reset env, one rollout can contain
            # data from different levels, which is very suboptimal for GAE




@dataclass
class SkillTracker:

    def __init__(self, levels: list[LevelType]):
        self._skills: dict[LevelType, Skill] = {lvl: Skill(lvl.value) for lvl in levels}

    def get_skill(self, level: LevelType):
        return self._skills[level]




@dataclass
class Skill:
    level_name: str
    difficulty: float = 0.0
    result_length: int = 10
    prev_results = np.zeros(result_length).tolist()
    episodes: int = 0
    regret: float = 0.0

    def update(self, regret: float):
        self.prev_results.pop(0)
        self.prev_results.append(regret)
        self.regret = np.mean(self.prev_results)

    def should_increase_difficulty(self, threshold) -> bool:
        return self.regret >= threshold


    