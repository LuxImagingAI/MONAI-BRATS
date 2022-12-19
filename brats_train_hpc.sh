#!/bin/bash -l
#SBATCH -J BRATS            # Display Name
#SBATCH -N 1                # Number of nodes
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -p gpu              # Job-type, can be batch (only CPU) or gpu
#SBATCH -G 1                # Number of GPUs for the job
#SBATCH --time=0-30:00:00   # Time limit
#SBATCH --array=0-4         # Definition of job array
#SBATCH --qos normal

epochs=50

nvidia-smi
export PATH="$HOME/miniconda/bin:$PATH"
source activate MONAI-BRATS
ulimit -n 2048

python brats_train.py --nfolds ${SLURM_ARRAY_TASK_COUNT} --fold ${SLURM_ARRAY_TASK_ID} --epochs $epochs

