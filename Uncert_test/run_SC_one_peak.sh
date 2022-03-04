#!/bin/bash

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128


source /lus/theta-fs0/software/thetagpu/conda/pt_master/2021-05-12/mconda3/setup.sh
mkdir one_peak_test
python uq.py  --peak "one_peak"  --n_points 4500 --json_file "One_peak.json"  --output_model "one_peak_test/"  --input_flag 0 --output_flag 1 --factor_reset 50 --n_iterations  100
source deactivate
