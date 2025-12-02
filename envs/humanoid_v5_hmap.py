import mujoco
from envs.humanoid_v5 import HumanoidEnv
from gymnasium.spaces import Box
import numpy as np

class HumanoidEnvHmap(HumanoidEnv):
    """Humanoid-v5 environment with heightmap observation added.
    Also it has additional methods for adapting terrain."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
        self.observation_space = Box(low, high, dtype=np.float32)

    def _local_to_world(self, local_points):
        pelvis_id = self.model.body('torso').id
        p = self.data.xpos[pelvis_id]          # pelvis world position
        R = self.data.xmat[pelvis_id].reshape(3, 3)  # rotation matrix

        # Rotate and translate
        return (R @ local_points.T).T + p
    
    def _get_heightmap(self):
        points_world = self._local_to_world(self.sample_points_local)
        heights = []
        # mask: only consider terrain group (group 1)
        geomgroup = np.zeros(6, dtype=np.uint8)
        geomgroup[1] = 1  
        
        for pw in points_world:
            # Ray starts slightly above point to avoid self-collision
            ray_start = np.array([pw[0], pw[1], pw[2] + 1.0])
            ray_dir = np.array([0, 0, -1.0])   # downward ray

            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                ray_start,
                ray_dir,
                geomgroup,
                1,      # flg_static: include static terrain
                -1,     # bodyexclude: disable
                geomid
            )

            if geomid[0] != -1:
                hit_z = ray_start[2] - dist
            else:
                hit_z = 0.0

            heights.append(hit_z)

        return np.array(heights, dtype=np.float32)
    
    def _get_obs(self):
        base_obs = super()._get_obs()
        heightmap = self._get_heightmap()

        return np.concatenate([base_obs, heightmap]).astype(np.float32)
