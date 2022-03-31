#!/bin/bash

if [[ -z $PUC_PROJ_ENV_SET ]]; then
    echo "Environment is not configured set"
    exit 255
fi

RCLONE_REMOTE=puc_data_bucket
REMOTE_PATH=files/raw_files

rclone sync --progress $RCLONE_REMOTE:$REMOTE_PATH/ $PROJ_ROOT/datasets/samples/raw/