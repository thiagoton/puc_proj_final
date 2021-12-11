#!/bin/sh
PROJECT_ROOT_FOLDER=`git rev-parse --show-toplevel`
tar -cvf $PROJECT_ROOT_FOLDER/datasets/samples/preprocessed.tar.xz $PROJECT_ROOT_FOLDER/datasets/samples/preprocessed