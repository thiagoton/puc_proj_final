#!/bin/bash
if [[ -z $PUC_PROJ_ENV_SET ]]; then
    echo "Environment is not configured set"
    exit -1
fi

RCLONE_REMOTE=puc_data_bucket
REMOTE_PATH=/files/raw_files/
DATASET_RAW_ROOT=$PROJ_ROOT/datasets/samples/raw

RAW_FILES[0]="$DATASET_RAW_ROOT/UrbanSound8K.tar.gz"

for file in ${RAW_FILES[@]};
do
    rclone sync --progress ${file} $RCLONE_REMOTE:$REMOTE_PATH
done