import os
from scripts import train


def main():
    run_dir = train.make_run_dir(train.PPO_CONFIG)

    # Content of the SLURM script
    content = f"""#!/bin/bash

#SBATCH --job-name=bipedal-walking-training-{int(train.PPO_CONFIG["timesteps"] // 1e6)}M
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./{run_dir}/logs/slurm.%j.%N.out
#SBATCH -e ./{run_dir}/logs/slurm.%x.%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={train.PPO_CONFIG["n_envs"]}

source ./.venv/bin/activate
python -u -m scripts.train {run_dir}
"""

    train_sh_path = os.path.join(".", "train.sh")
    with open(train_sh_path, "w") as f:
        f.write(content)
    # Make it executable
    print(f"train.sh written to {train_sh_path}")

if __name__ == "__main__":
    main()