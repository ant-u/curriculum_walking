import pickle
import numpy as np


def get_levels_with_succ_r(levels):
    for level in levels:
        level.regret = level.obstacles * (level.diff_stairs + level.diff_slab + level.diff_gap + level.diff_stump) / 4
        if level.succ_r == 0:
            level.regret = 0
    return levels

def get_difficultiest_level(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    levels = get_levels_with_succ_r(data)
    levels.sort(key=lambda x: x.regret)
    print(levels[-1])
    print(levels[-2])
    print(levels[-3])
    


get_difficultiest_level("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/logs/buffer/buffer_snapshot.pkl")