#/bin/sh

install_environment() {
    if ! command -v conda &> /dev/null
    then
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
        chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
        bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -f
    fi

    if ! conda env list | grep "puc_proj_final_env" -q
    then
        conda env create -f environment.yml
    else
        conda env update -n puc_proj_final_env -f environment.yml
    fi
}

if [[ ! -z $PUC_PROJ_ENV_SET ]]; then
    echo "Environment is already set"
    return
fi

install_environment

conda activate puc_proj_final_env

export PROJ_ROOT=`pwd`

# environment is ready
export PUC_PROJ_ENV_SET=1