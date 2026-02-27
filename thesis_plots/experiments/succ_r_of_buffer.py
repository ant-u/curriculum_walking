import re
from matplotlib import pyplot as plt
import numpy as np


def get_succ_rates_for_buffer(buffer_log_path):
    with open(buffer_log_path) as f:
        content = f.read()
    
    snapshots = re.split(r'(?=^None \* \d+)', content, flags=re.MULTILINE | re.IGNORECASE)
    buffer_succ_r = []
    for snapshot in snapshots:
        if not snapshot.strip():
            continue
        
        succ_r_values = re.findall(r'succ_r=([\d.]+)', snapshot)
        rates = [float(v) for v in succ_r_values]
        
        if rates:  # only add snapshots that have at least one non-None succ_r
            buffer_succ_r.append(rates)
    
    return buffer_succ_r


def create_succ_r_plot(buffer_log_path, save_path):
    succ_rates = get_succ_rates_for_buffer(buffer_log_path)

    avg_succ_rates = [100*sum(s)/len(s) for s in succ_rates]
    x = np.linspace(0, 300, len(avg_succ_rates))

    font_size_axis_ticks = 14
    font_size_title = 14
    font_size_label = 18
    font_size_legend = 14

    fig, ax1 = plt.subplots(figsize=(6, 5))
    color_1 = "steelblue"
    color_ep = "coral"

    ax1.set_xlabel("Buffer Updates", fontsize=font_size_label)
    ax1.set_ylabel("Difficulty (in %)", fontsize=font_size_label)  # color=color_reward

    color_1 = "steelblue"
    line1, = ax1.plot(x, avg_succ_rates, color=color_1, linewidth=2, label="dings")

    ax1.tick_params(axis="y", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    # ax1.set_ylim(bottom=-1, top=101)
    # ax1.set_xlim(left=0, right=len(x)*1.01)


    lines = [line1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=font_size_legend)
    # plt.title(f'Buffer Evolution', fontsize=font_size_title) #, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path)
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()



    


create_succ_r_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/buffer_succ_r_evolution_a.pdf")
create_succ_r_plot("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/buffer_succ_r_evolution_b.pdf")