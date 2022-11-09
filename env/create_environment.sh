#!/bin/bash

module load lang/Anaconda3
conda env create -f environment.yml
conda activate MONAI-BRATS
python3 -m ipykernel install --user --name MONAI-BRATS

