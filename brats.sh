#!/bin/bash -l
#SBATCH -J BRATS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128                # Cores assigned to each tasks
#SBATCH --time=0-6:00:00
#SBATCH -p batch
#SBATCH --qos normal

conda activate MONAI-BRATS
python brats_train.py
