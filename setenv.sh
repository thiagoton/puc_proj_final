#/bin/sh

install_environment() {
    if ! command -v conda &> /dev/null
    then
        CONDA_PREFIX=$(realpath ~/miniconda3)
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
        chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
        bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -f -p $CONDA_PREFIX
    fi

    # select appropriate env for gpu/cpu
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[LOG] Setting CPU environment"
        CONDA_ENV_FILE="environment.yml"
    else
        echo "[LOG] Setting GPU environment"
        CONDA_ENV_FILE="environment-gpu.yml"
    fi

    if ! conda env list | grep "puc_proj_final_env" -q
    then
        conda env create -f $CONDA_ENV_FILE
    else
        conda env update --satisfied-skip-solve -n puc_proj_final_env -f $CONDA_ENV_FILE
    fi
}

if [[ ! -z $PUC_PROJ_ENV_SET ]]; then
    echo "Environment is already set"
    return
fi

install_environment
conda activate puc_proj_final_env
CONDA_PREFIX=$(conda info --base)

# fix: dvc failed to set cache type
mkdir -p datasets/samples

# unlock our rclone configuration
SECRET_KEY_FILE=".local/secret.key"
if [[ ! -f "$SECRET_KEY_FILE" ]]; then
    echo "[ERR] $SECRET_KEY_FILE does not exist. Please, provide a valid secret file in order to continue"
    return
fi

git crypt unlock $SECRET_KEY_FILE
RCLONE_CONF=$(realpath rclone.conf)
if ! mount | grep -q .dvc_cache -q; then
    rm -rf .dvc_cache
    if [[ -z $DVC_CACHE_PATH ]]; then
        echo -n "[LOG] Mounting cache..."
        mkdir -p .dvc_cache
        rclone --config $RCLONE_CONF mount --daemon --vfs-cache-mode full puc_data_bucket:/files/dvc_cache .dvc_cache
        if [[ $? -eq 0 ]]; then
            echo "OK"
        else
            echo "[ERR] Failed mounting cache"
            return
        fi
    else
        echo "[LOG] Using local cache from '$DVC_CACHE_PATH'"
        ln -s "$DVC_CACHE_PATH" .dvc_cache
    fi
else
    echo "[LOG] Cache is already mounted"
fi

PROJ_ROOT=$(pwd)
export PROJ_ROOT="$PROJ_ROOT"
export CONDA_PREFIX="$CONDA_PREFIX"
export RCLONE_CONF="$RCLONE_CONF"

# environment is ready
export PUC_PROJ_ENV_SET=1

echo "[LOG] Finished. Let's rock!!!"