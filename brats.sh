#!/bin/bash -l
#SBATCH -J BRATS
#SBATCH -N 1
#SBATCH --ntasks-per-node=7
#SBATCH -c 1                # Cores assigned to each tasks
#SBATCH -G 1
#SBATCH --time=0-8:00:00
#SBATCH -p gpu
#SBATCH --qos normal

#module load lang/Anaconda3/2020.11
module load system/CUDA
sleep 2s
conda activate MONAI-BRATS

nvidia-smi
python brats_train.py
monai-deploy package brats_deploy.py --tag brats_app --output-dir "" --model "model/model.ts"
docker save brats_app > brats_app.tar
