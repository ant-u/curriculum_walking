#!/bin/bash

#SBATCH --job-name=cassie-training-5
#SBATCH --comment="Training for robotic RL bipedal locomotion."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Ant.Utz@campus.lmu.de
#SBATCH -D ./
#SBATCH -o ./slurm.%j.%N.out
#SBATCH -e ./slurm.%x.%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --ntasks=1

source ./.venv/bin/activate
python -u -m scripts.train
