#!/bin/bash

#SBATCH --job-name=bipedal-walking-training-15M
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260116-190322/logs/slurm.%j.%N.out
#SBATCH -e ./runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260116-190322/logs/slurm.%x.%j.err
#SBATCH --partition=Krater
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ./.venv/bin/activate
python -u -m scripts.train -p runs/humanoidenvcurr_ppo_lr1e-04_seed0_20260116-190322 -m "moved geoms to worldbody, creating with contype=conaffinity=1, disabling with script. fast training?"
