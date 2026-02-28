
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scripts.util.plot import Plot, CleanDoublePlot


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
    y_prog = np.zeros(len(x))
    y_succ = np.zeros(len(x))
    all_data.pop(5)
    
    for timestep in range(0, len(x)):
        obst_avg = 0
        succ_r_avg = 0
        for level in range(0, len(all_data)):
            obst_avg += np.mean(all_data[level]['total_runs'][timestep])
            if level == 0:
                succ_r_avg += all_data[level]['succ_r'][timestep] * 2
            else:
                succ_r_avg += all_data[level]['succ_r'][timestep]
        obst_avg /= len(all_data)
        y_prog[timestep] = obst_avg
        succ_r_avg /= len(all_data)
        y_succ[timestep] = succ_r_avg
    return x, y_prog, y_succ
    # plot = Plot("General Obstacle Capability", "Time Steps", "Progress", "avg. prog. on eval levels")
    # plot.update_with_x(y, x)
    # plot.save(os.path.join(base_path, "eval", "general_cabability.svg"))
    # print(f"y max: {max(y)}")
    # print(f"timestep: {x[np.where(y == max(y))[0][0]]}")


def make_single_plot(base_path):
    x, y_prog, y_succ = create_general_cap_plot(base_path)
    plot = CleanDoublePlot(x, y_prog, y_succ, f"General Obstacle Cap, y_max={round(max(y_prog),5)}, at timestep {x[np.where(y_prog == max(y_prog))[0][0]]}", "prog", "succ_r", "Time steps", "Progress")

    # plot = Plot(f"General Obstacle Cap, y_max={round(max(y_prog),5)}, at timestep {x[np.where(y_prog == max(y_prog))[0][0]]}", "Time Steps", "Progress", "avg. prog. on eval levels")
    # plot.update_with_x(y_prog, x)
    save_path = os.path.join(base_path, "eval", "general_cabability.svg")
    os.makedirs(os.path.join(base_path, "eval"), exist_ok=True)
    plot.save(save_path)
    print(f"progress y max: {max(y_prog)}")
    print(f"progress timestep: {x[np.where(y_prog == max(y_prog))[0][0]]}")
    print(f"succ_r y max: {max(y_succ)}")
    print(f"succ_r timestep: {x[np.where(y_succ == max(y_succ))[0][0]]}")
    a = y_succ.copy()
    a.sort()
    for b in range(1, 6):
        print(x[np.where(y_succ == a[-b])[0][0]])



def create_double_plot(base_path1, base_path2, save_path):
    x1, y1, _ = create_general_cap_plot(base_path1)
    x2, y2, _ = create_general_cap_plot(base_path2)

    if len(y1) != len(y2):
        longer_one = y1 if len(y1) > len(y2) else y2
        longer_one_x = x1 if len(x1) > len(x2) else x2
        shorter_one = y2 if longer_one is y1 else y1
        difference = len(longer_one) - len(shorter_one)
        longer_one = longer_one[:-difference]
        longer_one_x = longer_one_x[:-difference]

    x = np.array(longer_one_x) / 1_000_000
    y_1_adapted = np.array(longer_one) * 100
    y_2_adapted = np.array(shorter_one) * 100
    plot = CleanDoublePlot(x, y_1_adapted, y_2_adapted, "", #"General Obstacle Capability"
                           "Experiment A", "Experiment B", "Training environment steps (in M)", "Progress (in %)",
                           x_lim=[0, 300], y_lim=[0, 20], legend_pos="lower center", color_1="salmon", color_2="green")
    print(f"y1 max: {max(y1)}")
    print(f"timestep: {x1[np.where(y1 == max(y1))[0][0]]}")
    print(f"y2 max: {max(y2)}")
    print(f"timestep: {x2[np.where(y2 == max(y2))[0][0]]}")
    plot.save(save_path)





# create_general_cap_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125")
# create_general_cap_plot("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206")

# make_single_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-160853")
# make_single_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154150")
# make_single_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231")

# make_single_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100512")
# make_single_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161329")
# make_single_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154623")

# make_single_plot("runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618")
# make_single_plot("runs/result_exp_e/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155353")

# make_single_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350")
# make_single_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120621")
# make_single_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161918")
make_single_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-155601")


# create_double_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125", 
                #    "runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206",
                #    "thesis_plots/experiments/gen_capab_a_and_b.pdf")