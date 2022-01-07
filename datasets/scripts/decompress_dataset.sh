#!/bin/sh
PROJECT_ROOT_FOLDER=`git rev-parse --show-toplevel`
mkdir -p $PROJECT_ROOT_FOLDER/datasets/samples
tar -xvf $PROJECT_ROOT_FOLDER/datasets/preprocessed.tar.xz -C $PROJECT_ROOT_FOLDER/datasets/samples