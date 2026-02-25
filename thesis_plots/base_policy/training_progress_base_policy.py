import re
from matplotlib import pyplot as plt
import numpy as np

def parse_rl_log(file_path):
    mean_ep_lengths = []
    mean_rewards = []
    total_timesteps = []

    # Pattern to match a full eval block
    block_pattern = re.compile(
        r'\|\s*eval/\s*\|[^\n]*\n'
        r'\|\s*mean_ep_length\s*\|\s*([\d.e+]+)\s*\|\n'
        r'\|\s*mean_reward\s*\|\s*([-\d.e+]+)\s*\|\n'
        r'\|\s*time/\s*\|[^\n]*\n'
        r'\|\s*total_timesteps\s*\|\s*([\d.e+]+)\s*',
        re.IGNORECASE
    )

    with open(file_path, 'r') as f:
        content = f.read()

    for match in block_pattern.finditer(content):
        ep_len = float(match.group(1))
        reward  = float(match.group(2))
        steps   = float(match.group(3))

        mean_ep_lengths.append(ep_len)
        mean_rewards.append(reward)
        total_timesteps.append(round(steps / 1_000_000, 2))  # rounded to millions

    return (
        np.array(mean_ep_lengths),
        np.array(mean_rewards),
        np.array(total_timesteps)
    )

def plot_training_progress(file_path):
    ep_lengths, rewards, timesteps = parse_rl_log(file_path)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # --- Left y-axis: Mean Reward ---
    color_reward = "steelblue"
    ax1.set_xlabel("Training Environment Steps (in M)", fontsize=12)
    ax1.set_ylabel("Mean Reward", color=color_reward, fontsize=12)
    line1, = ax1.plot(timesteps, rewards, color=color_reward, linewidth=2, label="Mean Reward")
    ax1.tick_params(axis="y", labelcolor=color_reward)
    ax1.set_ylim(bottom=0, top=21000)
    ax1.set_xlim(left=0, right=50)

    # --- Right y-axis: Mean Episode Length ---
    color_ep = "coral"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Episode Length", color=color_ep, fontsize=12)
    line2, = ax2.plot(timesteps, ep_lengths, color=color_ep, linewidth=2, linestyle="--", label="Mean Episode Length")
    ax2.tick_params(axis="y", labelcolor=color_ep)
    ax2.set_ylim(bottom=0, top=2100)

    # --- Legend, title, layout ---
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=10)

    plt.title("Training Progress of Base Policy", fontsize=14) #, fontweight="bold")
    fig.tight_layout()
    # plt.savefig("training_progress.png", dpi=150)
    plt.savefig("thesis/plots/training_progress_base_policy.pdf")
    plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()
    

# ep_lengths, rewards, timesteps = parse_rl_log("runs/base_lidar_gait_height_resistant/logs/slurm.79912.krater07.out")
plot_training_progress("runs/base_lidar_gait_height_resistant/logs/slurm.79912.krater07.out")
# plot = DoubleLinePlot("Training progress of base policy", "Time steps (in M)", "")
