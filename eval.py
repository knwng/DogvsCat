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

slim = tf.contrib.slim

ap = argparse.ArgumentParser()
ap.add_argument('--val_dataset', type=common.readable_directory, required=True)
ap.add_argument('--train_dir', type=common.readable_directory, required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--epoch', default=1, type=common.positive_int)
ap.add_argument('--batch_size', default=32, type=common.positive_int)
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
                                                 shuffle=True,
                                                 num_epochs=num_epochs,
                                                 capacity=batch_size)
    reader = tf.TextLineReader()
    key, value = reader.read(dataqueue)
    dir_, label = tf.decode_csv(records=value, record_defaults=[["string"], ["string"]], field_delim=" ")
    label = tf.one_hot(tf.string_to_number(label, tf.int32), depth=2, dtype=tf.float32)
    imagecontent = tf.read_file(dir_)
    image = tf.image.decode_png(imagecontent, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [image_size, image_size])
    image = (image - rgb_mean) / 128.
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=6,
                                                        capacity=3 * batch_size+500, min_after_dequeue=500)
    return image_batch, label_batch 

val_image, val_label = get_batch([args.val_dataset], args.epoch, args.batch_size, _RGB_MEAN, args.image_size)

with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
    images = tf.placeholder(tf.float32, (None, args.image_size, args.image_size, 3))
    labels = tf.placeholder(tf.float32, (None, 2))
    net, endpoints = inception_resnet_v2(images, is_training=True, create_aux_logits=False, num_classes=None)
    net = slim.flatten(endpoints['Conv2d_7b_1x1'])
    net = slim.dropout(net, 0.8, is_training=True, scope='Dropout2')
    logits = tf.contrib.layers.fully_connected(net, 2, activation_fn=None, scope='Logits2')
    predictions = tf.nn.softmax(logits, name='Predictions2')
global_step = tf.Variable(0, name='global_step', trainable=False)
metric_acc, op_acc = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1))
metric_precision, op_precision = tf.metrics.precision(tf.argmax(labels, 1), tf.argmax(predictions, 1))
metric_recall, op_recall = tf.metrics.recall(tf.argmax(labels, 1), tf.argmax(predictions, 1))
ckpt_saver = tf.train.Saver()

with tf.Session() as sess:
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
    _step = 0
    try:
        while not coord.should_stop():
            _images, _labels = sess.run([val_image, val_label])
            sess.run([op_acc, op_precision, op_recall], feed_dict={images: _images, labels: _labels})
            _step += 1
            tf.logging.info("Evaluating Step: {}".format(_step))
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished')
    finally:
        _acc, _precision, _recall = sess.run([metric_acc, metric_precision, metric_recall])
        elapsed_time = time.time() - start_time
        tf.logging.info('*'*40)
        tf.logging.info("Validation Finished | Time: {}".format(elapsed_time))
        tf.logging.info("Accuracy: {}".format(_acc))
        tf.logging.info("Precision: {}".format(_precision))
        tf.logging.info("Recall: {}".format(_recall))
        tf.logging.info('*'*40)
        coord.request_stop()
    coord.join(threads)
    sess.close()
