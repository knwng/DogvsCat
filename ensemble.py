#!/bin/env python
import os
import numpy as np
import csv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--fold_num', type=int, default=10)
ap.add_argument('--ensemble_root')
ap.add_argument('--submission_name')

args = ap.parse_args()

results_fn = open('ensembled_submission.csv', 'w')
results_writer = csv.writer(results_fn)

results = []
for i in range(args.fold_num):
    print('Reading Submission {}'.format(i))
    csv_fn = open(os.path.join('{}-{}'.format(args.ensemble_root, i + 1), args.submission_name), 'r')
    csv_reader = csv.reader(csv_fn)
    for line in csv_reader:
        if csv_reader.line_num == 1:
            submission_head = line
            continue
        if i == 0:
            results.append([int(line[0]), [float(line[1])]])
        else:
            results[int(line[0])-1][1].append(float(line[1]))
    csv_fn.close()
print('Ensembling...')
# ensembled_results = map(lambda x:[x[0], max(x[1]) if max(x[1]) > 0.5 else min(x[1])], results)
# ensembled_results = map(lambda x:[x[0], max(x[1])], results)
ensembled_results = map(lambda x:[x[0], np.mean(x[1])], results)
print('Writing results into csv...')
results_writer.writerow(submission_head)
results_writer.writerows(ensembled_results)
    
