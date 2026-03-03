import pickle
from matplotlib import pyplot as plt
import numpy as np

def correct_plots(path, save_path, color):
    with open(path, "rb") as f:
        data = pickle.load(f)

    x = np.arange(0, 11) * 0.1
    x_inices = [int(xs*100) for xs in x]
    y_values = np.array(data["y"])
    y = y_values[x_inices]
    x *= 100


    font_size_axis_ticks = 14
    font_size_title = 14
    font_size_label = 18
    font_size_legend = 14

    fig, ax1 = plt.subplots(figsize=(6, 5))
    # color_reward = "steelblue"
    # color_ep = "coral"

    ax1.set_xlabel("Success Rate (in %)", fontsize=font_size_label)
    ax1.set_ylabel("Ratio (in %)", fontsize=font_size_label)  # color=color_reward

    line1, = ax1.plot(x, y, color=color, linewidth=2, marker="o", label="Success rate")

    ax1.tick_params(axis="y", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.set_ylim(bottom=0, top=100)
    ax1.set_xlim(left=0, right=100)


    lines = [line1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=font_size_legend)
    # plt.title(f'Buffer Evolution', fontsize=font_size_title) #, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path)
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()



correct_plots("runs/base_lidar_gait_height_resistant/eval/init_buffer_level_dump1_c.pkl", "thesis_plots/experiments/plots/initial_buffer_diff_harder.pdf", "salmon")
correct_plots("runs/base_lidar_gait_height_resistant/eval/init_buffer_level_dump2_c.pkl", "thesis_plots/experiments/plots/initial_buffer_diff_easier.pdf", "green")