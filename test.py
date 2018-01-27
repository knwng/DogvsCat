#!/bin/env python
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os
import random
import numpy as np
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import argparse
import common
import time
import cv2
import logging
import csv

slim = tf.contrib.slim

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ap = argparse.ArgumentParser()
ap.add_argument('--test_dataset', type=common.readable_directory, required=True)
ap.add_argument('--train_dir', type=common.readable_directory, required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--epoch', default=1, type=common.positive_int)
ap.add_argument('--batch_size', default=200, type=common.positive_int)
ap.add_argument('--image_size', default=224, type=common.positive_int)

args = ap.parse_args()

_RGB_MEAN = [123.68, 116.78, 103.94]

log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.train_dir, '{}_{}.log'.format('validation_log', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))))
fh.setLevel(logging.INFO)
log.addHandler(fh)

def get_batch(content, num_epochs, batch_size, rgb_mean, image_size):
    dataqueue = tf.train.string_input_producer(content,
                                                 shuffle=False,
                                                 num_epochs=num_epochs,
                                                 capacity=batch_size)
    reader = tf.TextLineReader()
    key, value = reader.read(dataqueue)
    dir_ = value
    imagecontent = tf.read_file(dir_)
    image = tf.image.decode_png(imagecontent, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [image_size, image_size])
    image = (image - rgb_mean) / 128.
    image_batch, fn_batch = tf.train.batch([image, dir_], batch_size=batch_size, num_threads=1,
                                           capacity=3 * batch_size+500, 
                                           allow_smaller_final_batch=True)
    return image_batch, fn_batch 

test_image, test_fn = get_batch([args.test_dataset], args.epoch, args.batch_size, _RGB_MEAN, args.image_size)

with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
    images = tf.placeholder(tf.float32, (None, args.image_size, args.image_size, 3))
    net, endpoints = inception_resnet_v2(images, is_training=True, create_aux_logits=False, num_classes=None)
    net = slim.flatten(endpoints['Conv2d_7b_1x1'])
    net = slim.dropout(net, 0.8, is_training=True, scope='Dropout2')
    logits = tf.contrib.layers.fully_connected(net, 2, activation_fn=None, scope='Logits2')
    predictions = tf.nn.softmax(logits, name='Predictions2')
global_step = tf.Variable(0, name='global_step', trainable=False)
ckpt_saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    ckpt_dir = os.path.join(args.train_dir, args.checkpoint)
    saver.restore(sess, ckpt_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)
    
    submission_fn = open(os.path.join(args.train_dir, 'submission_{}.csv'.format(args.checkpoint)),'w')
    sub_writer = csv.writer(submission_fn)
    sub_writer.writerow(['id', 'label'])

    start_time = time.time()
    counter = 0
    try:
        while not coord.should_stop():
            _images, _fns = sess.run([test_image, test_fn])
            _predictions = sess.run(predictions, feed_dict={images: _images})
            for i in range(len(_fns)):
                sub_writer.writerow([_fns[i].split('/')[-1].split('.')[0], round(_predictions[i][0], 2)])
                counter += 1
            print('{} images processed'.format(counter))
            submission_fn.flush()
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished')
    finally:
        elapsed_time = time.time() - start_time
        submission_fn.close()
        coord.request_stop()
    coord.join(threads)
    sess.close()
