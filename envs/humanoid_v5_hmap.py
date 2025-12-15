import os
from envs.humanoid_v5 import HumanoidEnv
from gymnasium.spaces import Box
import numpy as np
from envs.levels import get_step_level


class HumanoidEnvHmap(HumanoidEnv):
    """Humanoid-v5 environment with heightmap observation added.
    Also it has additional methods for adapting terrain."""
    def __init__(self, **kwargs):
        path = os.path.abspath("./models/humanoid_hmap.xml")
        super().__init__(xml_file=path, **kwargs)
        
        # self.num_points_x = 6      # forward sampling
        # self.num_points_y = 5      # sideways sampling
        # self.y_width = 1.5         # meters left/right of pelvis
        # self.x_forward = 4       # meters in front of pelvis
        # self.x_start = -1         # skip the immediate area directly below

        # xs = np.linspace(self.x_start, self.x_forward, self.num_points_x)
        # ys = np.linspace(-self.y_width, self.y_width, self.num_points_y)

        # grid = []
        # for y in ys:
        #     for x in xs:
        #         grid.append([x, y, 0])  
        # self.sample_points_local = np.array(grid)
        # height_map_dim = len(self.sample_points_local)
        # assert height_map_dim == len(self.data.site_xpos), "Number of observation points does NOT match "\
        # "number of site markers in humanoid_hmap.xml. Make sure there are exactly as much points in xml than "\
        # "defined in HumanoidEnvHmap. (num_points_x * num_points_y must be same as len(site_markers))"

        # low = np.concatenate([self.observation_space.low,[-np.inf]*height_map_dim])
        # high = np.concatenate([self.observation_space.high,[np.inf]*height_map_dim])
        # self.observation_space = Box(low, high, dtype=np.float64)

    # def _local_to_world(self, local_points):
    #     pelvis_id = self.model.body('torso').id
    #     p = self.data.xpos[pelvis_id]          # pelvis world position
    #     R = self.data.xmat[pelvis_id].reshape(3, 3)  # rotation matrix

    #     # Rotate and translate
    #     return (R @ local_points.T).T + p
    
    # def _get_heightmap(self):
    #     points_world = self._local_to_world(self.sample_points_local)
    #     heights = []

    #     for i, pw in enumerate(points_world):
    #         # Get height directly from height field
    #         hfield_id = 0
    #         x_idx = self._get_hfield_index(pw[0], x=True, y=False)
    #         y_idx = self._get_hfield_index(pw[1], x=False, y=True)
            
    #         # Ensure indices are within bounds
    #         x_idx = max(0, min(x_idx, self.model.hfield_ncol[hfield_id]-1))
    #         y_idx = max(0, min(y_idx, self.model.hfield_nrow[hfield_id]-1))
            
    #         # Get height from height field data
    #         height = self.model.hfield_data[y_idx * self.model.hfield_ncol[hfield_id] + x_idx]
    #         heights.append(height)
            
    #         self.data.site_xpos[i] = [pw[0], pw[1], height]  # updating pos of sites
        
    #     return np.array(heights, dtype=np.float32)
    
    def _get_obs(self):
        base_obs = super()._get_obs()
        # heightmap = self._get_heightmap()

        # return np.concatenate([base_obs, heightmap]).astype(np.float32)
        return base_obs

    # def set_env_level_stairs(self, height, x_ratio, y_ratio):
    #     self.model.hfield_data = get_step_level(self.model, height, x_ratio, y_ratio)

    # def reset_model(self):
    #     ret = super().reset_model()
    #     self.init_qpos[0] = -8
    #     return ret

    # def _get_hfield_index(self, pos, x: bool, y: bool):
    #     """Map coord space  to hfield space.
    #     So either one of the two, with the vars as defined in hfield in humanoid_hmap.xml:
    #     - [-x_size, x_size] --> [0, ncol]
    #     - [-y_size, y_size] --> [0, nrow]

    #     x and y have to be exclusively true
    #     """
    #     hfield_id = 0
    #     if x and not y:
    #         coord_upper = self.model.hfield_size[hfield_id][0]
    #         hfield_upper = self.model.hfield_ncol[hfield_id]
    #     if y and not x:
    #         coord_upper = self.model.hfield_size[hfield_id][1]
    #         hfield_upper = self.model.hfield_nrow[hfield_id]

    #     normalized = (pos - (-coord_upper)) / (2 * coord_upper)  # transform into normalized space [0..1]
    #     transformed = normalized * (hfield_upper)  # map to hfield space [0..nrow] or [0..ncol]
    #     return int(transformed)
