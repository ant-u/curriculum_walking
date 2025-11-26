import os
import sys
from datetime import datetime


def main():
    # Check argument count
    
    #TODO: adapt for new architecture
    if len(sys.argv) != 3:
        print("Usage: python create_train_script.py <steps in M> <name extension>")
        sys.exit(1)

    steps = int(sys.argv[1])
    name_extension = sys.argv[2]
    steps_total = steps * 1_000_000
    now = datetime.now()
    dir_name = f"{now.month:02d}_{now.day:02d}__{now.hour:02d}_{steps}M_{name_extension}"
    checkpoints_dir = os.path.join(".", "checkpoints", dir_name)
    
    # Create the checkpoints directory (if it doesn't exist)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Created or verified folder: {checkpoints_dir}")

    # Path to train.sh
    train_sh_path = os.path.join(".", "train.sh")

    # Content of the SLURM script
    content = f"""#!/bin/bash

#SBATCH --job-name=cassie-training-{steps}
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./checkpoints/{dir_name}/slurm.%j.%N.out
#SBATCH -e ./checkpoints/{dir_name}/slurm.%x.%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --ntasks=1

source ./.venv/bin/activate
python -u -m src.train_main -n {dir_name} -s {steps_total} -t -l 1
"""

    # Write the file
    with open(train_sh_path, "w") as f:
        f.write(content)

    # Make it executable
    print(f"train.sh written to {train_sh_path}")

if __name__ == "__main__":
    main()