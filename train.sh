#!/bin/bash

#SBATCH --job-name=bipedal-walking-training-20M
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260110-002402/logs/slurm.%j.%N.out
#SBATCH -e ./runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260110-002402/logs/slurm.%x.%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ./.venv/bin/activate
python -u -m scripts.train -p runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260110-002402 -m "20M training with default xml file to find out what more training time does"
