import numpy as np

class PerformaneEstimator():
    """Evaluates performance of Agent in currenct env/level.
    This is crucial for CurriculumManager for further training planing.
    """
    
    def __init__(self) -> None:
        pass
    
    def estimate(self, advantages):
        adv = np.array(advantages)
        neg_advs = adv[adv < 0]
        sum_neg_advs = np.sum(neg_advs)
        # print(advantages)
        return 0