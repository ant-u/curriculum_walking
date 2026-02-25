from dataclasses import dataclass
import pickle
from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

@dataclass
class BufferLevel():
    """Representing a level based on params and seed, used for buffer. 
    With all params and seed given, level can be identically reconstruced by LevelGenerator.
    Also, level regret and succ_rate are saved here since needed in Buffer."""
    seed: int
    obstacles: float
    diff_slab: float
    diff_stairs: float
    diff_stump: float
    diff_gap: float
    regret: Optional[float] = None
    succ_r: Optional[float] = None
    learnability: Optional[float] = None


class EvalLevel(BufferLevel):
    def __init__(self, seed, obstacles, diff_slab, diff_stairs, diff_stump, diff_gap,
                 regret=None, succ_r=None, learnability=None, progress=None):
        super().__init__(seed, obstacles, diff_slab, diff_stairs, diff_stump, diff_gap, regret, succ_r, learnability)
        self.progress = progress


def correct_cvar_plot(path):
    with open(path, "rb") as f: 
        data = pickle.load(f)
    levels = data['levels']
    alphas = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    prog_levels = levels.copy()
    levels.sort(key=lambda x: x.succ_r)
    prog_levels.sort(key=lambda x: x.progress)

    x = np.array(alphas) / 100
    y_succ_r = np.zeros(len(x))
    y_prog = np.zeros(len(x))
    for i, alpha in enumerate(x):
        n_elems = int(alpha * len(levels))
        elems_succ = levels[:n_elems]
        elems_prog = prog_levels[:n_elems]
        mean_succ_r = None
        mean_prog = None
        if len(elems_succ) > 0:
            mean_succ_r = np.mean([elem.succ_r for elem in elems_succ])
        if len(elems_prog) > 0:
            mean_prog = np.mean([elem.progress for elem in elems_prog])
        y_succ_r[i] = mean_succ_r
        y_prog[i] = mean_prog

    x_positions = range(len(alphas))
    y_succ_r *= 100
    y_prog *= 100

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_reward = "steelblue"
    ax1.set_xlabel(r"$\alpha$ (in %)", fontsize=12)
    ax1.set_ylabel("Ratio (in %)", fontsize=12)  # color=color_reward
    line1, = ax1.plot(x_positions, y_prog, color=color_reward, linewidth=2, label="Mean Progress")
    # ax1.tick_params(axis="y", labelcolor=color_reward)
    # ax1.set_ylim(bottom=-0.01, top=0.2)
    # ax1.set_xlim(left=-0.5, right=13)

    color_ep = "coral"
    line2, = ax1.plot(x_positions, y_succ_r, color=color_ep, linewidth=2, marker='o', label="Mean Success Rate")  #linestyle="--"
    # ax2.tick_params(axis="y", labelcolor=color_ep)
    # ax2.set_ylim(bottom=0, top=2100)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    plt.title(f'CVaR evaluation of Base Policy with N={len(levels)}', fontsize=14) #, fontweight="bold")
    ax1.set_xticks(ticks=x_positions, labels=alphas)
    # ax1.set_xlim(min(x), max(x))
    # ax1.set_xscale('log')
    # ax1.xaxis.set_major_locator(FixedLocator(x))
    # ax1.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x]))
    fig.tight_layout()
    plt.savefig("thesis_plots/base_policy/cvar_base_policy.pdf")
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()

correct_cvar_plot("runs/base_lidar_gait_height_resistant/eval/cvar_buffer_dump.pkl")