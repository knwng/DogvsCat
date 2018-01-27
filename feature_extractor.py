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
import scipy

slim = tf.contrib.slim

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ap = argparse.ArgumentParser()
ap.add_argument('--test_dataset', type=common.readable_directory, required=True)
ap.add_argument('--train_dir', type=common.readable_directory, required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--epoch', default=1, type=common.positive_int)
ap.add_argument('--batch_size', default=1, type=common.positive_int)
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

    start_time = time.time()
    counter = 0
    try:
        while not coord.should_stop():
            _images, _fns = sess.run([test_image, test_fn])
            _input_features, _bottleneck_features = sess.run([endpoints['Conv2d_1a_3x3'], endpoints['Conv2d_3b_1x1']], feed_dict={images: _images})
            _input_feature_enlarge = scipy.ndimage.interpolation.zoom(_input_features, (1, 2, 2, 1))
            _bottleneck_features_enlarge = scipy.ndimage.interpolation.zoom(_bottleneck_features, (1, 4, 4, 1))
            print('Input Feature: {}'.format(_input_feature_enlarge.shape))
            print('Bottleneck Feature: {}'.format(_bottleneck_features_enlarge.shape))
            print('File Name: {}'.format(_fns))
            _input_list = np.split(_input_feature_enlarge[0], 32, 2)
            _bottleneck_list = np.split(_bottleneck_features_enlarge[0], 80, 2)
            for i in range(len(_input_list)):
                _max = max(_input_list[i].flatten())
                _min = min(_input_list[i].flatten())
                normed = 255 * ((_input_list[i] - _min) / (_max - _min) )
                print('Data | max: {} | min: {}'.format(max(normed.flatten()), min(normed.flatten())))
                cv2.imwrite(os.path.join('tmp', 'Input_Feature-{}.jpg'.format(i)), normed.astype(np.uint8))
            for i in range(len(_bottleneck_list)):
                _max = max(_bottleneck_list[i].flatten())
                _min = min(_bottleneck_list[i].flatten())
                normed = 255 * ((_bottleneck_list[i] - _min) / (_max - _min) )
                cv2.imwrite(os.path.join('tmp', 'Bottleneck_Feature-{}.jpg'.format(i)), normed.astype(np.uint8))
            origin_img = cv2.imread(_fns[0])
            origin_img = cv2.resize(origin_img, (224, 224))
            cv2.imwrite(os.path.join('tmp', 'origin_img.jpg'), origin_img)
            break
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished')
    finally:
        elapsed_time = time.time() - start_time
        coord.request_stop()
    coord.join(threads)
    sess.close()
