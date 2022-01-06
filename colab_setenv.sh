#!/bin/bash
set -e

install_environment() {
    if ! command -v conda &> /dev/null
    then
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
        chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
        bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -f -p /usr/local
    fi

    if ! conda env list | grep "puc_proj_final_env" -q
    then
        conda env create -f environment.yml
    fi
}

install_environment

echo "Environment has been set"
