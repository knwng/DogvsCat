#!/bin/env python
import os
import argparse
import random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--labelmap', default='./label_map.txt')
args = ap.parse_args()

random.seed()
image_dir = 'test'
image_list = os.listdir(image_dir)
image_list = sorted(image_list, key=lambda x:int(x.split('/')[-1].split('.')[0]))
# random.shuffle(image_list)

with open('test_submission.txt', 'w') as f:
    label_dict = {}
    with open(args.labelmap, 'r') as fl:
        for line in fl:
            label_dict[line.split(' ')[0]] = line.split(' ')[1]
    count = 0
    for line in image_list:
        f.write('{}\n'.format(os.path.join('dataset', 'dogsvscats', image_dir, line)))
        count += 1
        print('Create test dataset: {}'.format(count))

'''
if args.data_split_type == 'random-80-20':
    train_fn = 'train.txt'
    test_fn = 'test.txt'
    with open(train_fn, 'w') as ftrain, open(test_fn, 'w') as ftest:
        label_dict = {}
        with open(args.labelmap, 'r') as fl:
            for line in fl:
                label_dict[line.split(' ')[0]] = line.split(' ')[1]
        count = 0
        for line in image_list:
            choice = random.random()
            if choice > 0.2:
                ftrain.write('{} {}'.format(os.path.join('dataset', 'dogsvscats', image_dir, line), label_dict[line[0:3]]))
            else:
                ftest.write('{} {}'.format(os.path.join('dataset', 'dogsvscats', image_dir, line), label_dict[line[0:3]]))
            count += 1
            print('Spliting image {} into {} fold | totally {}'.format(line, 'train' if choice > 0.2 else 'test', count))
else:
    folds = np.split(np.array(image_list), args.fold_num)
    for i in range(args.fold_num):
        print('Processing Fold {}'.format(i))
        train_fn = '{}-fold-train-{}.txt'.format(args.fold_num, i+1)
        test_fn = '{}-fold-test-{}.txt'.format(args.fold_num, i+1)
        with open(train_fn, 'w') as ftrain, open(test_fn, 'w') as ftest:
            label_dict = {}
            with open(args.labelmap, 'r') as fl:
                for line in fl:
                    label_dict[line.split(' ')[0]] = line.split(' ')[1]
            count = 0
            for j in range(args.fold_num):
                if j == i:
                    for line in folds[j]:
                        ftest.write('{} {}'.format(os.path.join('dataset', 'dogsvscats', image_dir, line), label_dict[line[0:3]]))
                        count += 1
                else:
                    for line in folds[j]:
                        ftest.write('{} {}'.format(os.path.join('dataset', 'dogsvscats', image_dir, line), label_dict[line[0:3]]))
                        count += 1
                print('Processing Fold {} | {} images processed'.format(i, count))

'''
