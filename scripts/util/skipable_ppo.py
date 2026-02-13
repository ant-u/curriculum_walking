from stable_baselines3 import PPO

class SkippablePPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_training = False
     
    def train(self):
        if self.skip_training:
            self.skip_training = False
            return
        super().train()