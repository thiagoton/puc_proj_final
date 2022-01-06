#!/bin/bash
set -e

conda activate puc_proj_final_env 
dvc pull
dvc exp run