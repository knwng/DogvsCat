#!/bin/bash

GPU_NUM=$1
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
FOLD=$2

echo "Using No.${GPU_NUM} | Training ${FOLD}th fold"
source ./venv/bin/activate
python ./train_dataset.py \
    --train_dataset "./dataset/assign/10-fold-train-${FOLD}.txt"  \
    --val_dataset "./dataset/assign/10-fold-test-${FOLD}.txt"  \
    --train_dir "./experiments/expr_assign_10-fold-${FOLD}"  \
    --epoch 20  \
    --learning_rate 1e-4  \
    --batch_size 32
