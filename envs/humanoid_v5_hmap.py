import mujoco
import os
from envs.humanoid_v5 import HumanoidEnv
from gymnasium.spaces import Box
import numpy as np

from envs.levels import get_step_level

class HumanoidEnvHmap(HumanoidEnv):
    """Humanoid-v5 environment with heightmap observation added.
    Also it has additional methods for adapting terrain."""
    def __init__(self, **kwargs):
        path = os.path.abspath("./models/humanoid.xml")
        super().__init__(xml_file=path, **kwargs)
        
        self.num_points_x = 7      # sideways sampling
        self.num_points_y = 5      # forward sampling
        self.x_width = 0.4         # meters left/right of pelvis
        self.y_forward = 1.2       # meters in front of pelvis
        self.y_start = 0.1         # skip the immediate area directly below

        xs = np.linspace(-self.x_width, self.x_width, self.num_points_x)
        ys = np.linspace(self.y_start, self.y_forward, self.num_points_y)

        grid = []
        for y in ys:
            for x in xs:
                grid.append([x, y, 0])  
        self.sample_points_local = np.array(grid)
        height_map_dim = len(self.sample_points_local)
        low = np.concatenate([self.observation_space.low,-np.ones(height_map_dim) * 5.0])
        high = np.concatenate([self.observation_space.high,np.ones(height_map_dim) * 5.0])
        self.observation_space = Box(low, high, dtype=np.float64)

    def _local_to_world(self, local_points):
        pelvis_id = self.model.body('torso').id
        p = self.data.xpos[pelvis_id]          # pelvis world position
        R = self.data.xmat[pelvis_id].reshape(3, 3)  # rotation matrix

        # Rotate and translate
        return (R @ local_points.T).T + p
    
    def _get_heightmap(self):
        points_world = self._local_to_world(self.sample_points_local)
        heights = []

        for pw in points_world:
            # Get height directly from height field
            hfield_id = 0
            x_idx = int((pw[0] + self.model.hfield_size[hfield_id][0]/2) * 
                    self.model.hfield_ncol[hfield_id] / self.model.hfield_size[hfield_id][0])
            y_idx = int((pw[1] + self.model.hfield_size[hfield_id][1]/2) * 
                    self.model.hfield_nrow[hfield_id] / self.model.hfield_size[hfield_id][1])
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, self.model.hfield_ncol[hfield_id]-1))
            y_idx = max(0, min(y_idx, self.model.hfield_nrow[hfield_id]-1))
            
            # Get height from height field data
            height = self.model.hfield_data[y_idx * self.model.hfield_ncol[hfield_id] + x_idx]
            heights.append(height)
        
        return np.array(heights, dtype=np.float32)
    
    def _get_obs(self):
        base_obs = super()._get_obs()
        heightmap = self._get_heightmap()

        return np.concatenate([base_obs, heightmap]).astype(np.float32)
    
    def set_env_level_stairs(self, height, x_ratio, y_ratio):
        self.model.hfield_data = get_step_level(self.model, height, x_ratio, y_ratio)
