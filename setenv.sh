#/bin/sh

install_environment() {
    if ! command -v conda &> /dev/null
    then
        CONDA_PREFIX=`realpath ~/miniconda3`
        wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
        chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
        bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -f -p $CONDA_PREFIX
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
CONDA_PREFIX=`conda info --base`

conda activate puc_proj_final_env

# unlock our rclone configuration
git crypt unlock .local/secret.key
RCLONE_CONF=`realpath rclone.conf`
if ! `mount | grep -q .dvc_cache -q`; then
    echo -n "[LOG] Mounting cache..."
    rclone --config $RCLONE_CONF mount --daemon --vfs-cache-mode full puc_data_bucket:/files/dvc_cache .dvc_cache
    if [[ $? -eq 0 ]]; then
        echo "OK"
    else
        echo "[ERR] Failed mounting cache"
        return
    fi
else
    echo "[LOG] Cache is already mounted"
fi

export PROJ_ROOT=`pwd`
export CONDA_PREFIX=$CONDA_PREFIX
export RCLONE_CONF=$RCLONE_CONF

# environment is ready
export PUC_PROJ_ENV_SET=1

echo "[LOG] Finished. Let's rock!!!"