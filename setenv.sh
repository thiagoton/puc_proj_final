#! /bin/bash

install_environment() {
    if ! command -v conda &> /dev/null; then
        CONDA_PREFIX=$(realpath ~/miniconda3)
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
        chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
        bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -f -p "$CONDA_PREFIX"
    else
        CONDA_PREFIX=$(conda info --base)
    fi

    # select appropriate env for gpu/cpu
    if [[ -n $COLAB_GPU ]]; then
        echo "[LOG] Setting Colab environment"
        CONDA_ENV_FILE="environment-colab.yml"
    elif ! command -v nvidia-smi &> /dev/null; then
        echo "[LOG] Setting CPU environment"
        CONDA_ENV_FILE="environment.yml"
    else
        echo "[LOG] Setting GPU environment"
        CONDA_ENV_FILE="environment-gpu.yml"
    fi

    source "$CONDA_PREFIX/etc/profile.d/conda.sh"

    if ! conda env list | grep "puc_proj_final_env" -q
    then
        conda env create -f $CONDA_ENV_FILE
    else
        conda env update -n puc_proj_final_env -f $CONDA_ENV_FILE
    fi
}

if [[ -n $PUC_PROJ_ENV_SET ]]; then
    echo "Environment is already set"
    return
fi

# install only it not at colab environment
if [[ -z $COLAB_GPU ]]; then
    install_environment
    conda activate puc_proj_final_env
fi

PROJ_ROOT=$(pwd)
export PROJ_ROOT="$PROJ_ROOT"
export CONDA_PREFIX="$CONDA_PREFIX"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

# environment is ready
export PUC_PROJ_ENV_SET=1

echo "[LOG] Finished. Let's rock!!!"