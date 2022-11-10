#!/bin/bash

pip install conda
conda env create -f environment.yml
conda activate MONAI-BRATS
python3 -m ipykernel install --user --name MONAI-BRATS

