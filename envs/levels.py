from typing import Tuple
import numpy as np


def set_slab(model, data, x_ratio, height):
    activate_shape(model, "slab_shape")
    geom = model.geom("slab_shape")
    floor = model.geom("floor")
    mocap_id = model.body("slab").mocapid[0]

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    move_mocap_pos(data, mocap_id, x_pos, 0, height - geom.size[2])


def unset_slab(model, data):
    deactivate_shape(model, "slab_shape")
    mocap_id = model.body("slab").mocapid[0]
    move_mocap_pos(data, mocap_id, 10, 40, 0)


def set_stairs(model, data, x_ratio, step_length, step_height):
    """Set level to statis, step_length must be in [0;5], step_height in [0;1]"""
    sanity_check = 0 <= step_length <= 5 and 0 <= step_height <= 1
    assert sanity_check, "values out of bound: steplength is [0;5] and stepheight is [0;1]."
    geom_obj = []
    for i in range(1, 9):
        activate_shape(model, f"stairs_shape_{i}")
        geom_obj.append(model.geom(f"stairs_shape_{i}"))
    floor_obj = model.geom("floor")
    mocap_id = model.body("stairs").mocapid[0]


    x_pos = (floor_obj.size[0] * 2 * x_ratio) - floor_obj.size[0] + geom_obj[0].size[0]
    move_mocap_pos(data, mocap_id, x_pos, 0, 0)
    for i, o in enumerate(geom_obj):
        object_height = o.size[2]
        model.geom_pos[o.id][0] = i * step_length
        model.geom_pos[o.id][2] = step_height - object_height + i*step_height


def unset_stairs(model, data):
    mocap_id = model.body("stairs").mocapid[0]
    for i in range(1,9):
        deactivate_shape(model, f"stairs_shape_{i}")
    move_mocap_pos(data, mocap_id, 10, 40, 0)


def set_log(model, data, x_ratio, height, size):
    """Set a log to lie in the way. height is z of middlepoint, size is diameter of log.
    For limiting max height of log: height + size <= 1."""
    assert height + size <= 1, "Log height and size are too high, unvalid obstacle."
    activate_shape(model, "log_shape")
    geom = model.geom("log_shape")
    floor = model.geom("floor")
    mocap_id = model.body("log").mocapid[0]

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    move_mocap_pos(data, mocap_id, x_pos, 0, height - geom.size[2])
    model.geom_size[geom.id][0] = size


def unset_log(model, data):
    deactivate_shape(model, "log_shape")
    mocap_id = model.body("log").mocapid[0]
    move_mocap_pos(data, mocap_id, 10, 40, 0)


def set_stump(model, data, x_ratio, height, depth):
    """Set a stump as level. Obstacle is meant as a stump in the way to conquer.
    Depth is absolute. Height and depth both have to be <= 1.
    Slab is for stepping up and walking on."""
    assert height <= 1 and depth <= 1, "Height or depth are to high, unvalid obstacle."
    activate_shape(model, "stump_shape")
    geom = model.geom("stump_shape")
    floor = model.geom("floor")
    mocap_id = model.body("stump").mocapid[0]

    x_pos = (floor.size[0] * 2 * x_ratio) - floor.size[0] + geom.size[0]
    move_mocap_pos(data, mocap_id, x_pos, 0, height - geom.size[2])
    model.geom_size[geom.id][0] = depth / 2


def unset_stump(model, data):
    deactivate_shape(model, "stump_shape")
    mocap_id = model.body("stump").mocapid[0]
    move_mocap_pos(data, mocap_id, 10, 40, 0)


def move_mocap_pos(data, id, x, y, z):
    data.mocap_pos[id][0] = x
    data.mocap_pos[id][1] = y
    data.mocap_pos[id][2] = z


def activate_shape(model, name):
    gid = model.geom(name).id
    model.geom_rgba[gid][3] = 1.0
    model.geom_contype[gid] = 1
    model.geom_conaffinity[gid] = 1


def deactivate_shape(model, name):
    gid = model.geom(name).id
    model.geom_rgba[gid][3] = 0.0
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0


def reset_all_levels(model, data):
    unset_slab(model, data)
    unset_stairs(model, data)
    unset_log(model, data)
    unset_stump(model, data)
