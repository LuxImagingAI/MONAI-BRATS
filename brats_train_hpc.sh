#!/bin/bash -l
#SBATCH -J BRATS            # Display Name
#SBATCH -N 1                # Number of nodes
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -p gpu              # Job-type, can be batch (only CPU) or gpu
#SBATCH -G 1                # Number of GPUs for the job
#SBATCH --time=0-30:00:00   # Time limit
#SBATCH --array=0-3         # Definition of job array
#SBATCH --qos normal

epochs=100

conda activate MONAI-BRATS
nvidia-smi
python brats_train.py --nfolds ${SLURM_ARRAY_TASK_COUNT} --fold ${SLURM_ARRAY_TASK_ID} --epochs $epochs


