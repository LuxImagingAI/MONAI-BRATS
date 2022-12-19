#!/bin/bash -l
#SBATCH -J BRATS            # Display Name
#SBATCH -N 1                # Number of nodes
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -p gpu              # Job-type, can be batch (only CPU) or gpu
#SBATCH -G 1                # Number of GPUs for the job
#SBATCH --time=0-4:00:00   # Time limit
#SBATCH --qos normal

sleep 2s

nvidia-smi
export PATH="$HOME/miniconda/bin:$PATH"
source activate MONAI-BRATS

python brats_deploy.py --input "data/Task01_BrainTumour/imagesTr" --output "output/labels" --model "output/models"


