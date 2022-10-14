#!/bin/bash -l
#SBATCH -J BRATS
#SBATCH -N 1
#SBATCH --ntasks-per-node=7
#SBATCH -c 1                # Cores assigned to each tasks
#SBATCH -G 1
#SBATCH --time=0-30:00:00
#SBATCH -p gpu
#SBATCH --qos normal
#SBATCH --array=0-4

epochs=100

#module load lang/Anaconda3/2020.11
module load system/CUDA
sleep 2s
conda activate MONAI-BRATS

nvidia-smi
python brats_train.py --nfolds ${SLURM_ARRAY_TASK_COUNT} --fold ${SLURM_ARRAY_TASK_ID} --epochs $epochs
python brats_deploy.py --input "data/Task01_BrainTumour/imagesTs" --output "output" --model "model"

