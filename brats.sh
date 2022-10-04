#!/bin/bash -l
#SBATCH -J BRATS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -G 1
#SBATCH --time=0-8:00:00
#SBATCH -p gpu
#SBATCH --qos normal

conda activate MONAI-BRATS
python brats_train.py
