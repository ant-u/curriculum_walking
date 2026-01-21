from typing import Tuple
import numpy as np
import mujoco


def set_slab(model, x_ratio, height):
    activate_shape(model, "slab_shape")
    geom = model.geom("slab_shape")
    floor = model.geom("floor")

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    model.geom_pos[geom.id] = [x_pos, 0, height - geom.size[2]]


def unset_slab(model):
    deactivate_shape(model, "slab_shape")
    shape_id = model.geom("slab_shape").id
    model.geom_pos[shape_id] = [10, 40, 0]


def set_stairs(model, x_ratio, step_length, step_height):
    """Set level to statis, step_length must be in [0;5], step_height in [0;1]"""
    sanity_check = 0 <= step_length <= 5 and 0 <= step_height <= 1
    assert sanity_check, "values out of bound: steplength is [0;5] and stepheight is [0;1]."
    geom_obj = []
    for i in range(1, 9):
        activate_shape(model, f"stairs_shape_{i}")
        geom_obj.append(model.geom(f"stairs_shape_{i}"))
    floor_obj = model.geom("floor")

    x_pos = (floor_obj.size[0] * 2 * x_ratio) - floor_obj.size[0] + geom_obj[0].size[0]
    for i, o in enumerate(geom_obj):
        object_height = o.size[2]
        model.geom_pos[o.id][0] = i * step_length + x_pos
        model.geom_pos[o.id][1] = 0
        model.geom_pos[o.id][2] = step_height - object_height + i*step_height


def unset_stairs(model):
    for i in range(1,9):
        shape_name = f"stairs_shape_{i}"
        shape_id = model.geom(shape_name).id
        model.geom_pos[shape_id] = [10, 40, 0]
        deactivate_shape(model, shape_name)


def set_log(model, x_ratio, height, size):
    """Set a log to lie in the way. height is z of middlepoint, size is diameter of log.
    For limiting max height of log: height + size <= 1."""
    assert height + size <= 1, "Log height and size are too high, unvalid obstacle."
    activate_shape(model, "log_shape")
    geom = model.geom("log_shape")
    floor = model.geom("floor")

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    model.geom_pos[geom.id] = [x_pos, 0, height - geom.size[2]]
    model.geom_size[geom.id][0] = size


def unset_log(model):
    deactivate_shape(model, "log_shape")
    shape_id = model.geom("log_shape").id
    model.geom_pos[shape_id] = [10, 40, 0]


def set_stump(model, x_ratio, height, depth):
    """Set a stump as level. Obstacle is meant as a stump in the way to conquer.
    Depth is absolute. Height and depth both have to be <= 1.
    Slab is for stepping up and walking on."""
    assert height <= 1 and depth <= 1, "Height or depth are to high, unvalid obstacle."
    activate_shape(model, "stump_shape")
    geom = model.geom("stump_shape")
    floor = model.geom("floor")

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    model.geom_pos[geom.id] = [x_pos, 0, height - geom.size[2]]
    model.geom_size[geom.id][0] = depth / 2


def unset_stump(model):
    deactivate_shape(model, "stump_shape")
    geom_id = model.geom("stump_shape").id
    model.geom_pos[geom_id] = [10, 40, 0]


def set_ramp(model, x_ratio, angle):
    """Set a ramp as level. ramp is always 10m long, beginning and angle can be adjusted."""
    activate_shape(model, "ramp_shape")
    geom = model.geom("ramp_shape")
    floor = model.geom("floor")

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    model.geom_pos[geom.id] = [x_pos, 0, 0 - geom.size[2]]
    yaw_rad = np.deg2rad(-angle)
    euler = [0, yaw_rad, 0]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, euler, "xyz")
    model.geom_quat[geom.id] = quat


def unset_ramp(model):
    deactivate_shape(model, "ramp_shape")
    geom_id = model.geom("stump_shape").id
    model.geom_pos[geom_id] = [10, 40, 0]


def activate_shape(model, name):
    gid = model.geom(name).id
    model.geom_rgba[gid][3] = 1.0
    model.geom_contype[gid] = 1
    model.geom_conaffinity[gid] = 1
    model.geom_condim[gid] = 3


def deactivate_shape(model, name):
    gid = model.geom(name).id
    model.geom_rgba[gid][3] = 0.0
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0
    model.geom_condim[gid] = 1


def reset_all_levels(model):
    unset_slab(model)
    unset_stairs(model)
    unset_log(model)
    unset_stump(model)
    unset_ramp(model)


def set_condim_all_geoms(model, condim: int):
    geom_names = ["slab_shape", "log_shape", "stump_shape", "ramp_shape"]
    for i in range(1, 9):
        geom_names.append(f"stairs_shape_{i}")
    for name in geom_names:
        id = model.geom(name).id
        model.geom_condim[id] = condim
