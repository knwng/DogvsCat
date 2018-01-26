#!/bin/bash

GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}

source ./venv/bin/activate

FOLD=$2 # 6th finished

CHECKPOINT=$3

echo "Using GPU ${GPU} | Generate Submission for fold ${FOLD}"

./test.py \
    --test_dataset  ./dataset/dogsvscats/test_submission.txt \
    --train_dir ./experiments/expr_assign_10-fold-${FOLD} \
    --checkpoint model-${CHECKPOINT} \
    --batch_size 32
