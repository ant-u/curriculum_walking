from envs.curriculum.level_generator import LevelDescription


class GeomHandler:
    def __init__(self):
        self.x_gap = 1e-3
        self.z_gap = 1e-3

    def activate_shape(self, model, name):
        gid = model.geom(name).id
        model.geom_rgba[gid][3] = 1.0
        model.geom_contype[gid] = 1
        model.geom_conaffinity[gid] = 1
        model.geom_condim[gid] = 3

    def deactivate_shape(self, model, name):
        gid = model.geom(name).id
        model.geom_rgba[gid][3] = 0.0
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_condim[gid] = 1

    def activate_default_platform(self, model):
        self.activate_shape(model, "platform_middle")
        model.geom_pos[model.geom("platform_middle").id][2] = -2.5

    def deactivate_default_platform(self, model):
        self.deactivate_shape(model, "platform_middle")
        model.geom_pos[model.geom("platform_middle").id][2] = -10

    def deactivate_all_elements(self, model):
        for i in range((model.geom_group == 2).sum()):
            self.deactivate_shape(model, f"element_{i}")

    def deactivate_all_stumps(self, model):
        for i in range(10):
            self.deactivate_shape(model, f"stump_{i}")
            model.geom_pos[model.geom(f"stump_{i}").id][2] = -7

    def set_no_level(self, model):
        self.deactivate_all_elements(model)
        self.deactivate_all_stumps(model)
        self.activate_default_platform(model)
                
    def set_flat_level(self, model):
        self.deactivate_default_platform(model)
        x_offset = model.geom_size[model.geom("platform_start").id][0] + model.geom_size[model.geom("element_0").id][0]
        n = (model.geom_group == 2).sum()
        for i in range(0,n):
            self.activate_shape(model, f"element_{i}")
            geom = model.geom(f"element_{i}")
            model.geom_pos[geom.id][0] = x_offset + i * (2 * model.geom_size[geom.id][0] + self.x_gap)
            model.geom_pos[geom.id][2] = (0.0 + self.z_gap) - model.geom_size[geom.id][2]
        end_platform = model.geom("platform_end")
        model.geom_pos[end_platform.id][0] = (model.geom_pos[geom.id][0] + 
            model.geom_size[geom.id][0] + self.x_gap + model.geom_size[end_platform.id][0])
        for i in range(0,10):
            model.geom_pos[model.geom(f"stump_{i}").id][2] = -7
        self.deactivate_all_stumps(model)

    def set_custom_level(self, model, level_des: LevelDescription):
        for i, e_height in enumerate(level_des.elements):
            geom = model.geom(f"element_{i}")
            model.geom_pos[geom.id][2] = e_height - geom.size[2] + self.z_gap
        end_platform = model.geom("platform_end")
        model.geom_pos[end_platform.id][2] = level_des.elements[-1] - model.geom_size[end_platform.id][2] + self.z_gap

        self.deactivate_all_stumps(model)
        for i, s_height in enumerate(level_des.stumps):
            if s_height is not None and i != 0:
                geom = model.geom(f"stump_{i}")
                self.activate_shape(model, f"stump_{i}")
                model.geom_pos[geom.id][2] = s_height - geom.size[2]
