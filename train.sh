#!/bin/bash

#SBATCH --job-name=bipedal-walking-training-10M
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./runs/humanoid-v5_ppo_lr1e-04_seed0_20251201-165335/slurm.%j.%N.out
#SBATCH -e ./runs/humanoid-v5_ppo_lr1e-04_seed0_20251201-165335/slurm.%x.%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ./.venv/bin/activate
python -u -m scripts.train runs/humanoid-v5_ppo_lr1e-04_seed0_20251201-165335
