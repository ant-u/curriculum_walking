
import os
import pickle
import numpy as np
from scripts.util.plot import Plot


def create_general_cap_plot(base_path):
    logs_path = os.path.join(base_path, "logs")

    sub_folders = ["easy", "extrem", "gaps", "hard", "medium", "plain", "slabs", "stairs", "stumps"]
    all_data = []
    for folder in sub_folders:
        pck_file = os.path.join(logs_path, folder, "eval_results.pkl")
        with open(pck_file, "rb") as f: 
            data = pickle.load(f)
        all_data.append(data)


    x = all_data[0]['time_steps']
    y = np.zeros(len(x))
    all_data.pop(5)
    
    for timestep in range(0, len(x)):
        obst_avg = 0
        for level in range(0, len(all_data)):
            obst_avg += np.mean(all_data[level]['total_runs'][timestep])
        obst_avg /= len(all_data)
        y[timestep] = obst_avg
    plot = Plot("General Obstacle Capability", "Time Steps", "Progress", "avg. prog. on eval levels")
    plot.update_with_x(y, x)
    plot.save(os.path.join(base_path, "eval", "general_cabability.svg"))
    print(f"y max: {max(y)}")
    print(f"timestep: {x[np.where(y == max(y))[0][0]]}")



# create_general_cap_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125")
create_general_cap_plot("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206")