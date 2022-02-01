#!/bin/bash
set -e

if [[ -z $PUC_PROJ_ENV_SET ]]
then
    echo "Environment is not set. Please run 'source setenv.sh' at the root dir"
    exit 1
fi

source $CONDA_PREFIX/etc/profile.d/conda.sh

conda activate puc_proj_final_env 
dvc pull
dvc exp run