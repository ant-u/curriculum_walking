import os
import mujoco
from envs.humanoid_base import HumanoidEnvBase
from gymnasium.spaces import Box
import numpy as np
import envs.levels as levels


class HumanoidEnvCurr(HumanoidEnvBase):
    """Humanoid-v5 environment with heightmap observation added.
    Also it has additional methods for adapting terrain."""

    def __init__(self, cnfg, xml_file: str | None = None, **kwargs):
        """xml file is xml file name under ./models/ which shall be loaded."""

        if xml_file == None:
            path = os.path.abspath(f"./models/{cnfg["xml_file"]}")
        else:
            path = os.path.abspath(f"./models/{xml_file}")
        super().__init__(xml_file=path, **kwargs)
        self.use_lidar = cnfg["use_lidar"]
        self.render_lidar = cnfg["render_lidar"]
        if self.render_lidar:
            assert self.use_lidar == True, "If render_lidar is True, use_lidar has to be true too."
            

        # TODO: make set levels survive the reset
        self.last_level = None
        self.level_kwargs = None

        if self.use_lidar:
            self.use_relative_height = cnfg["use_relative_height"]
            self.n_points_x = cnfg["n_points_x"]   # forward sampling
            self.n_points_y = cnfg["n_points_y"]   # sideways sampling
            self.y_width = cnfg["y_width"]         # meters left/right of pelvis
            self.x_forward = cnfg["x_forward"]     # meters in front of pelvis
            self.x_start = cnfg["x_start"]         # skip the immediate area directly below

            xs = np.linspace(self.x_start, self.x_forward, self.n_points_x)
            ys = np.linspace(-self.y_width, self.y_width, self.n_points_y)

            grid = []
            for y in ys:
                for x in xs:
                    grid.append([x, y, 0])  
            self.sample_points_local = np.array(grid)
            height_map_dim = len(self.sample_points_local)
            # NOTE: -3 because of 3 markers for coord system in plane middle
            assert height_map_dim == len(self.data.site_xpos[:-3]), "Number of observation points does NOT match "\
                "number of site markers in humanoid_hmap.xml. Make sure there are exactly as much points in xml "\
                "than defined in HumanoidEnvHmap. (num_points_x * num_points_y must be same as len(site_markers))"
            
            low = np.concatenate([self.observation_space.low,[-np.inf]*height_map_dim])
            high = np.concatenate([self.observation_space.high,[np.inf]*height_map_dim])
            self.observation_space = Box(low, high, dtype=np.float64)
        # self.set_env_level_slab(1, 0.66)
    
    def _get_obs(self):
        base_obs = super()._get_obs()
        if self.use_lidar:
            heightmap = self._get_heightmap()
            return np.concatenate([base_obs, heightmap]).astype(np.float32)
        return base_obs
    
    def _local_to_world(self, local_points):
        pelvis_id = self.model.body('torso').id
        p = self.data.xpos[pelvis_id]          # pelvis world position
        R = self.data.xmat[pelvis_id].reshape(3, 3)  # rotation matrix
        # Rotate and translate
        return (R @ local_points.T).T + p
    
    def _get_heightmap(self):
        # Convert local sample points to world coordinates
        world_points = self._local_to_world(self.sample_points_local)
        
        heights = np.zeros(len(world_points))
        torso_height = self.data.xpos[self.model.body('torso').id][2]
        torso_height = torso_height if self.use_relative_height == True else 0  # if not relative height: pick 0, results in absolute height
        
        for i, point in enumerate(world_points):
            # Ray-cast from above the point down to find terrain height
            # Start ray well above any expected terrain
            ray_start = np.array(point.copy())
            ray_start[2] = 100.0  # start 100m above
            
            ray_end = np.array(point.copy())
            ray_end[2] = -100.0  # end 100m below
            vec = ray_end - ray_start
            geomid = np.array([-1], dtype=np.int32)
            geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)  # only group 1, all obstacles (robot is 0)
            
            # Perform raycast
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                pnt=ray_start,
                vec=vec,
                geomgroup=geomgroup,
                flg_static=1,
                bodyexclude=self.model.body('torso').id,  # exclude robot body
                geomid=geomid
            )
            
            if geomid[0] >= 0 and distance >= 0:  # Hit something
                hit_point = ray_start + distance * vec
                absolute_point_height = hit_point[2]
            else:
                # No hit - assume ground plane at z=0
                absolute_point_height = 0.0
            heights[i] = absolute_point_height - torso_height  # if use_relative_height is false: torso height = 0 --> absolute height
            if self.render_lidar:
                self.data.site_xpos[i] = [point[0], point[1], absolute_point_height]  # updating pos of sites
        return heights

    def set_env_level_slab(self, height, x_ratio):
        levels.set_slab(self.model, self.data, x_ratio, height)
        self.last_level = self.set_env_level_slab
        self.level_kwargs = {"height": height, "x_ratio": x_ratio}

    def unset_env_level_slab(self):
        levels.unset_slab(self.model, self.data)
        self.last_level = self.level_kwargs = None

    def set_env_level_stairs(self, x_ratio, step_length, step_height):
        levels.set_stairs(self.model, self.data, x_ratio, step_length, step_height)
        self.last_level = self.set_env_level_stairs
        self.level_kwargs = {"x_ratio": x_ratio, 
                             "step_length": step_length, 
                             "step_height": step_height}
        
    def unset_env_level_stairs(self):
        levels.unset_stairs(self.model, self.data)
        self.last_level = self.level_kwargs = None

    def reset_model(self):
        ret = super().reset_model()
        # self.init_qpos[0] = -8
        if self.last_level and self.level_kwargs:
            self.last_level(**self.level_kwargs)
        return ret
