import os
from typing import List
import mujoco
from envs.humanoid_base import HumanoidEnvBase
from gymnasium.spaces import Box
import numpy as np
from envs.geom_handler import GeomHandler
from envs.curriculum.level_generator import Element, LevelDescription


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
        self.using_levels = cnfg["use_levels"] if "use_levels" in cnfg.keys() else True
        self.terminate_on_x = cnfg["terminate_at_x_border"] if "terminate_at_x_border" in cnfg.keys() else 0
        if self.render_lidar:
            assert self.use_lidar == True, "If render_lidar is True, use_lidar has to be true too."
        
        geom_h_z_gap = cnfg["geom_z_gap"]if hasattr(cnfg, "geom_z_gap") else 1e-3
        self.geom_handler = GeomHandler(geom_h_z_gap)
        self.current_level: LevelDescription = None
        if path.endswith("humanoid_plane.xml"):
            if self.using_levels:
                self.geom_handler.set_flat_level(self.model)
                self.change_level_flag = False
            else:
                self.geom_handler.set_no_level(self.model)

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
            if self.render_lidar:
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
            # heightmap = np.zeros(len(heightmap))
            base_obs = np.concatenate([base_obs, heightmap]).astype(np.float32)
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
            geomgroup = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)  # only group 1, all obstacles (robot is 0)
            
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

    def step(self, action):
        obs, reward, terminated, done, info  = super().step(action)
        x = info["x_position"]
        if self.terminate_on_x:
            if x >= self.terminate_on_x:
                reward += 100
                terminated = True
                info['success'] = True
                info['terminated_reason'] = 'goal_reached'
            else:
                info['sucess'] = False
        return obs, reward, terminated, done, info
    
    def set_level_template(self, level: LevelDescription):
        self.current_level = level
        self.change_level_flag = True

    def _create_level(self, level: LevelDescription):
        if level == None:
            self.geom_handler.set_flat_level(self.model)
        else:
            self.geom_handler.set_custom_level(self.model, level)

    def reset_model(self):
        if self.using_levels and self.change_level_flag:
            self._create_level(self.current_level)
            self.change_level_flag = False
        return super().reset_model()
    
    # TODO: add new is_healthy method to account for new height options
