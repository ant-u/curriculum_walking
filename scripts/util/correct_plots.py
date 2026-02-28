import os
import pickle
import numpy as np
from scripts.util.plot import DoubleLinePlot, FiveLinePlot


def main():
    base_path = "runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-232323"
    logs_path = os.path.join(base_path, "logs")

    sub_folders = ["easy", "extrem", "gaps", "hard", "medium", "plain", "slabs", "stairs", "stumps"]

    for folder in sub_folders:
        pck_file = os.path.join(logs_path, folder, "eval_results.pkl")
        with open(pck_file, "rb") as f: 
            data = pickle.load(f)

        total_runs = data["total_runs"]
        average_prog_total_runs = [np.clip(np.mean(x), 0, 1) for x in total_runs]
        x_data = np.array(data["time_steps"]) / 1e6
        plot = DoubleLinePlot(f"Evaluation of level {folder}", "time steps", "%", "Average progress", "Success rate", False)
        plot.update_with_x(average_prog_total_runs, data["succ_r"], x_data, 0)
        plot.save(os.path.join(logs_path, folder, "eval_plot.svg"))

    pck_file = os.path.join(logs_path, "buffer", "difficulty_ratios.pkl")
    with open(pck_file, "rb") as f: 
        data = pickle.load(f)
    difficulty_thresholds = [0.01, 0.02, 0.04, 0.1, 1]
    line_lables = [f"Diff under {x}" for x in difficulty_thresholds]
    plot = FiveLinePlot("Difficulty ratios", "Buffer updates", "Ratio in %", line_lables, False)
    plot_data = []
    for i in range(len(difficulty_thresholds)):
        plot_data.append(np.array(data)[:,i])
    plot.update(*plot_data, y_lim=0)
    plot.save(os.path.join(logs_path, "buffer", "difficulty_ratios.svg"))


main()
