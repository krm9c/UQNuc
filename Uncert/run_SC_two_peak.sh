#!/bin/bash

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

source /lus/theta-fs0/software/thetagpu/conda/pt_master/2021-05-12/mconda3/setup.sh
mkdir two_peak_test
python uq.py  --peak "two_peak"  --n_points 450000 --json_file "One_peak.json"  --output_model "two_peak_test/"  --input_flag 1 --output_flag 1 --factor_reset 10 --n_iterations  20
source deactivate
