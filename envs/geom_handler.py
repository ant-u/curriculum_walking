


class GeomHandler:
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

    def deactivate_all_obstacles(self, model):
        for i in range((model.geom_group == 2).sum()):
            self.deactivate_shape(model, f"obstacle_{i}")
                
    def init_obstacles(self, model):
        n = (model.geom_group == 2).sum()
        for i in range(0,n):
            self.activate_shape(model, f"obstacle_{i}")
