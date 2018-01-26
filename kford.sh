#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
FOLD_NUM=10

for ((i=1;i<=${FOLD_NUM};i++))
do
    echo "Training ${i}th fold"
    python ./train_dataset.py --train_dataset "./dataset/assign/10-fold-train-${i}.txt"  --val_dataset "./dataset/assign/10-fold-test-${i}.txt"  --train_dir "./experiments/expr_assign_10-fold"  --epoch 40
done
