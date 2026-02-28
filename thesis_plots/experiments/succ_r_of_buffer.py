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


def create_succ_r_plot(buffer_log_path, save_path, color_1="steelblue"):
    succ_rates = get_succ_rates_for_buffer(buffer_log_path)

    avg_succ_rates = [100*np.mean(s) for s in succ_rates]
    minimum = [100*min(s) for s in succ_rates]
    maximum = [100*max(s) for s in succ_rates]
    q25 = [100*np.quantile(s, 0.25) for s in succ_rates]
    q75 = [100*np.quantile(s, 0.75) for s in succ_rates]
    x = np.linspace(0, 300, len(avg_succ_rates))

    font_size_axis_ticks = 14
    font_size_title = 14
    font_size_label = 18
    font_size_legend = 14

    fig, ax1 = plt.subplots(figsize=(6, 5))

    ax1.set_xlabel("Training environment steps (in M)", fontsize=font_size_label)
    ax1.set_ylabel("Success rate (in %)", fontsize=font_size_label)  # color=color_reward

    line1, = ax1.plot(x, avg_succ_rates, color=color_1, linewidth=2, label="Success Rate IQR (25-75%)")
    ax1.fill_between(x, q25, q75, alpha=0.3, color=color_1)
    ax1.fill_between(x, minimum, maximum, alpha=0.1, color=color_1)

    ax1.tick_params(axis="y", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=font_size_axis_ticks) #labelcolor=color_reward
    # ax1.set_ylim(bottom=-1, top=101)
    # ax1.set_xlim(left=0, right=len(x)*1.01)


    lines = [line1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=font_size_legend)
    # plt.title(f'Buffer Evolution', fontsize=font_size_title) #, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path)
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()


# create_succ_r_plot("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/buffer_succ_r_evolution_a.pdf", "salmon")
# create_succ_r_plot("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/buffer_succ_r_evolution_b.pdf", "green")

# create_succ_r_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154150/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/plots/exp_c_buffer_succ_r_evol.pdf", "salmon")
# create_succ_r_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr5e-05_seed0_20260223-154623/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/plots/exp_d_buffer_succ_r_evol.pdf", "green")

create_succ_r_plot("runs/result_exp_c/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100231/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/plots/exp_c_2_buffer_succ_r_evol.pdf", "salmon")
create_succ_r_plot("runs/result_exp_d/humanoidenvcurr_ppo_lr1e-04_seed0_20260224-100512/logs/buffer/buffer_logs.txt", "thesis_plots/experiments/plots/exp_d_2_buffer_succ_r_evol.pdf", "green")