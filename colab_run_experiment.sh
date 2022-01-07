#!/bin/bash
set -e

source /usr/local/etc/profile.d/conda.sh

conda activate puc_proj_final_env 
dvc pull
dvc exp run