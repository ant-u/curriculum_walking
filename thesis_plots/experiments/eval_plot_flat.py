import pickle
from matplotlib import pyplot as plt
import numpy as np


def create_eval_plot_flat(path, name):
    with open(path, "rb") as f: 
        data = pickle.load(f)

    x = np.array(data['time_steps']) / 1000000
    y_succ_r = np.array(data['succ_r']) * 100
    y_prog = np.array([np.mean(prog) for prog in data['total_runs']]) * 100
    

    fig, ax1 = plt.subplots(figsize=(6, 5))
    color_reward = "steelblue"
    ax1.set_xlabel("Training environment steps (in M)", fontsize=20)
    ax1.set_ylabel("Ratio (in %)", fontsize=20)  # color=color_reward
    line1, = ax1.plot(x, y_prog, color=color_reward, linewidth=2, label="Mean Progress")
    ax1.tick_params(axis="y", labelsize=16) #labelcolor=color_reward
    ax1.tick_params(axis="x", labelsize=16) #labelcolor=color_reward
    ax1.set_ylim(bottom=-0.5, top=100.5)
    ax1.set_xlim(left=0, right=301)

    color_ep = "coral"
    line2, = ax1.plot(x, y_succ_r, color=color_ep, linewidth=2, label="Mean Success Rate")  #linestyle="--"
    # ax2.tick_params(axis="y", labelcolor=color_ep)
    # ax2.set_ylim(bottom=0, top=2100)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", fontsize=18)
    # plt.title('Flat level evaluation during curriculum training', fontsize=14) #, fontweight="bold")
    # ax1.set_xticks(ticks=x_positions, labels=alphas)
    # ax1.set_xlim(min(x), max(x))
    # ax1.set_xscale('log')
    # ax1.xaxis.set_major_locator(FixedLocator(x))
    # ax1.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x]))
    fig.tight_layout()
    plt.savefig(f"thesis_plots/experiments/{name}.pdf")
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()

create_eval_plot_flat("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/logs/plain/eval_results.pkl", "eval_plot_flat_exp_A")
create_eval_plot_flat("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/logs/plain/eval_results.pkl", "eval_plot_flat_exp_B")
