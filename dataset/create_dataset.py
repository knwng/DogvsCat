#!/bin/env python
import os
import argparse
import random
import numpy as np
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--is_train', action='store_true', default=True, 
        help='Specify training datset or not')
ap.add_argument('--data_split_type', choices=('random-80-20', 'k-fold'), default='random-80-20')
ap.add_argument('--fold_num', required=True, type=int)
ap.add_argument('--labelmap', default='./label_map.txt')
# ap.add_argument('--data_dir')
args = ap.parse_args()

# if args.data_split_type == 'k-fold' and 'fold_num' not in dir(args):
#     print 'No fold_num specified under k-fold strategy'
#     sys.exit()

random.seed()
image_dir = 'train' if args.is_train else 'test1'
image_list = os.listdir(image_dir)
print(image_list[:10])
random.shuffle(image_list)

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
    print('Fold shape: {}'.format(folds[0]))
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
                        ftrain.write('{} {}'.format(os.path.join('dataset', 'dogsvscats', image_dir, line), label_dict[line[0:3]]))
                        count += 1
                print('Processing Fold {} | {} images processed'.format(i, count))

