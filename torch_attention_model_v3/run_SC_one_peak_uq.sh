#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tork
python uq_v_One_UQ.py 
source deactivate
