#!/bin/bash

#SBATCH --job-name=bipedal-walking-cvar-eval
#SBATCH --comment="Evaluating CVAR for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./runs/cvar_logs/cvar_eval_slurm.%j.%N.out
#SBATCH -e ./runs/cvar_logs/slurm.%x.%j.err
#SBATCH --partition=Krater
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ./.venv/bin/activate
python -u -m scripts.cvar_eval