import re
from matplotlib import pyplot as plt
import numpy as np

def parse_rl_log(file_path):
    mean_ep_lengths = []
    mean_rewards = []
    total_timesteps = []
    success_rates = []

    # Pattern to match a full eval block
    block_pattern = re.compile(
        r'-+\s*Evaluating terrain:\s*plain\s*-+\n'
        r'.*?\n'  # Eval num_timesteps line
        r'.*?\n'  # Episode length line
        r'-+\n'
        r'\|\s*eval/\s*\|[^\n]*\n'
        r'\|\s*mean_ep_length\s*\|\s*([\d.e+]+)\s*\|\n'
        r'\|\s*mean_reward\s*\|\s*([-\d.e+]+)\s*\|\n'
        r'\|\s*time/\s*\|[^\n]*\n'
        r'\|\s*total_timesteps\s*\|\s*([\d.e+]+)\s*'
        r'(?:.*?)'                                           # skip remaining table lines (train/ etc.)
        r'Summary terrain plain:\s*success:\s*\d+\s*/\s*\d+\s*\((\d+\.\d+)\s*%\)',
        re.IGNORECASE | re.DOTALL
    )

    with open(file_path, 'r') as f:
        content = f.read()

    for match in block_pattern.finditer(content):
        ep_len = float(match.group(1))
        reward  = float(match.group(2))
        steps   = float(match.group(3))
        success_rate = float(match.group(4))


        mean_ep_lengths.append(ep_len)
        mean_rewards.append(reward)
        total_timesteps.append(round(steps / 1_000_000, 2))  # rounded to millions
        success_rates.append(success_rate)

    return (
        np.array(mean_ep_lengths),
        np.array(mean_rewards),
        np.array(total_timesteps),
        np.array(success_rates)
    )

def plot_training_progress(file_path):
    ep_lengths, rewards, timesteps, succ_rates = parse_rl_log(file_path)

    per_step_reward = np.array(rewards) / np.array(ep_lengths)
    best_policy_timesteps = timesteps[np.where(per_step_reward == max(per_step_reward[np.where(succ_rates == 100)]))]
    print(f"best per-step reward at a 100% run at timesteps: {best_policy_timesteps}")

    fig, ax1 = plt.subplots(figsize=(6, 5))
    

    # --- Left y-axis: Mean Reward ---
    color_reward = "steelblue"
    ax1.set_xlabel("Training Environment Steps (in M)", fontsize=12)
    ax1.set_ylabel("Mean Reward", color=color_reward, fontsize=12)
    line1, = ax1.plot(timesteps, rewards, color=color_reward, linewidth=2, label="Mean Reward")
    ax1.tick_params(axis="y", labelcolor=color_reward)
    # ax1.set_ylim(bottom=0, top=21000)
    # ax1.set_xlim(left=0, right=50)

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
    # plt.savefig("thesis/plots/training_progress_base_policy.pdf")
    # plt.savefig("thesis/plots/training_progress_base_policy.svg")
    plt.show()
    


# plot_training_progress("runs/base_lidar_gait_height_resistant/logs/slurm.79912.krater07.out")

plot_training_progress("runs/result_exp_a/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165125/logs/slurm.79905.krater08.out")
# plot_training_progress("runs/result_exp_b/humanoidenvcurr_ppo_lr1e-04_seed0_20260222-165206/logs/slurm.79906.krater09.out")
