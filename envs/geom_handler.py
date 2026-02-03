from envs.curriculum.level_generator import LevelDescription


class GeomHandler:
    def __init__(self, z_gap):
        self.z_gap = z_gap
        self.default_color = [1,1,1,1]

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
        model.geom_pos[model.geom("platform_middle").id][2] = -20

    def deactivate_all_elements(self, model):
        for i in range((model.geom_group == 2).sum()):
            self.deactivate_shape(model, f"element_{i}")
            geom = model.geom(f"element_{i}")
            model.geom_pos[geom.id][2] = -7.5 - model.geom_size[geom.id][2]

    def set_no_level(self, model):
        self.deactivate_all_elements(model)
        self.activate_default_platform(model)
                
    def set_flat_level(self, model):
        self.deactivate_default_platform(model)
        n = (model.geom_group == 2).sum()
        for i in range(0,n):
            self.activate_shape(model, f"element_{i}")
            geom = model.geom(f"element_{i}")
            model.geom_pos[geom.id][2] = (0.0 + self.z_gap) - model.geom_size[geom.id][2]
            model.geom_rgba[geom.id] = self.default_color
        
        # model.geom_pos[model.geom("element_10").id][2] = -3.95

    def set_custom_level(self, model, level_des: LevelDescription):
        for i, e_height in enumerate(level_des.elements):
            geom = model.geom(f"element_{i}")
            model.geom_pos[geom.id][2] = e_height - geom.size[2] + self.z_gap
            if level_des.types[i]:
                model.geom_rgba[geom.id][0:3] = level_des.types[i].value
            else:
                model.geom_rgba[geom.id] = self.default_color
        end_platform = model.geom("platform_end")
        model.geom_pos[end_platform.id][2] = level_des.elements[-1] - model.geom_size[end_platform.id][2] + self.z_gap
