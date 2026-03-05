import pickle
import os
import numpy as np
from matplotlib import pyplot as plt


def correct_buffer_plot(buffer_path, save_path):
    with open(os.path.join(buffer_path, "difficulty_ratios.pkl"), "rb") as f:
        data = np.array(pickle.load(f))

    fig, ax1 = plt.subplots(figsize=(6, 5))
    
    data *= 100
    x = np.arange(len(data))
    y_very_easy = data[:,0]
    y_easy = data[:,1]
    y_mid = data[:,2]
    y_hard = data[:,3]
    y_extreme = data[:,4]

    # --- Left y-axis: Mean Reward ---
    font_size_axis_ticks = 14
    font_size_title = 14
    font_size_label = 18
    font_size_legend = 14

    fig, ax1 = plt.subplots(figsize=(6, 5))
    color_reward = "steelblue"
    color_ep = "coral"

    ax1.set_xlabel("Buffer Updates", fontsize=font_size_label)
    ax1.set_ylabel("Difficulty (in %)", fontsize=font_size_label)  # color=color_reward

    colors = ["green", "olive", "orange", "red", "brown"]

    line1, = ax1.plot(x, y_very_easy, color=colors[0], linewidth=2, label="0 - 0,01")
    line2, = ax1.plot(x, y_easy, color=colors[1], linewidth=2, label="0,01 - 0,02")  #linestyle="--"
    line3, = ax1.plot(x, y_mid, color=colors[2], linewidth=2, label="0,02 - 0,04")  #linestyle="--"
    line4, = ax1.plot(x, y_hard, color=colors[3], linewidth=2, label="0,04 - 0,1")  #linestyle="--"
    line5, = ax1.plot(x, y_extreme, color=colors[4], linewidth=2, label="0,1 - 1,0")  #linestyle="--"

    ax1.tick_params(axis="y", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.set_ylim(bottom=-1, top=101)
    ax1.set_xlim(left=0, right=len(x)*1.01)


    lines = [line1, line2, line3, line4, line5]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=font_size_legend)
    # plt.title(f'Buffer Evolution', fontsize=font_size_title) #, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path)
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()


# correct_buffer_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/logs/buffer", "thesis_plots/experiments/buffer_difficulty_ratios_exp_a.pdf")
# correct_buffer_plot("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/logs/buffer", "thesis_plots/experiments/buffer_difficulty_ratios_exp_b.pdf")

# correct_buffer_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154150/logs/buffer", "thesis_plots/experiments/plots/exp_c_buffer_diff_ratios.pdf")
# correct_buffer_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154623/logs/buffer", "thesis_plots/experiments/plots/exp_d_buffer_diff_ratios.pdf")

# correct_buffer_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231/logs/buffer", "thesis_plots/experiments/plots/exp_c_2_buffer_diff_ratios.pdf")
# correct_buffer_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100512/logs/buffer", "thesis_plots/experiments/plots/exp_d_2_buffer_diff_ratios.pdf")

# correct_buffer_plot("runs/result_exp_e/humanoidenvcurr_ppo_lr3e-05_seed0_20260223-161618/logs/buffer", "thesis_plots/experiments/plots/exp_e_buffer_diff_ratios.pdf")
correct_buffer_plot("runs/result_exp_e/humanoidenvcurr_ppo_lr1e-04_seed0_20260301-103659/logs/buffer", "thesis_plots/experiments/plots/exp_e_buffer_diff_ratios.pdf")
# correct_buffer_plot("runs/result_exp_f/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-120350/logs/buffer", "thesis_plots/experiments/plots/exp_f_buffer_diff_ratios.pdf")