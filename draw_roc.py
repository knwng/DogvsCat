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
ap.add_argument('--val_dataset', type=common.readable_directory, required=True)
ap.add_argument('--train_dir', type=common.readable_directory, required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--learning_rate', default=1e-4, type=common.positive_float)
ap.add_argument('--epoch', default=1, type=common.positive_int)
ap.add_argument('--batch_size', default=200, type=common.positive_int)
ap.add_argument('--image_size', default=224, type=common.positive_int)

args = ap.parse_args()

_RGB_MEAN = [123.68, 116.78, 103.94]

val_fn = args.val_dataset

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
    # value = train_queue.dequeue()
    reader = tf.TextLineReader()
    key, value = reader.read(dataqueue)
    dir_, label = tf.decode_csv(records=value, record_defaults=[["string"], ["string"]], field_delim=" ")
    label = tf.one_hot(tf.string_to_number(label, tf.int32), depth=2, dtype=tf.float32)
    imagecontent = tf.read_file(dir_)
    image = tf.image.decode_png(imagecontent, channels=3)
    # image = tf.image.decode_jpeg(imagecontent, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [image_size, image_size])
    image = (image - rgb_mean) / 128.
    # image = image - _RGB_MEAN
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
model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'InceptionResnetV2')
global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(
        learning_rate=args.learning_rate,
        global_step=global_step,
        decay_steps=1000,
        decay_rate=0.96, 
        staircase=True)
loss = tf.losses.softmax_cross_entropy(labels, logits)
metric_acc, op_acc = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1))
metric_precision, op_precision = tf.metrics.precision(tf.argmax(labels, 1), tf.argmax(predictions, 1))
metric_recall, op_recall = tf.metrics.recall(tf.argmax(labels, 1), tf.argmax(predictions, 1))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
ckpt_saver = tf.train.Saver(max_to_keep=0)

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # last_checkpoint = tf.train.latest_checkpoint(args.train_dir)
    ckpt_dir = os.path.join(args.train_dir, args.checkpoint)
    saver.restore(sess, ckpt_dir)
    # tf.logging.info('Resume training from [{}]'.format(ckpt_dir))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)

    roc_fn = open('roc_data_1.csv', 'w')
    roc_writer = csv.writer(roc_fn)

    _step = 0
    try:
        while not coord.should_stop():
            _images, _labels = sess.run([val_image, val_label])
            _predictions = sess.run(predictions, feed_dict={images: _images, labels: _labels})
            for i in range(len(_labels)):
                # roc_writer.writerow([np.argmax(_labels[i]), _predictions[i][np.argmax(_labels[i])]])
                roc_writer.writerow([np.argmax(_labels[i]), _predictions[i][1]])
            roc_fn.flush()
            _step += 1
            tf.logging.info("Evaluating Step: {}".format(_step))
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished')
    finally:
        roc_fn.close()
        coord.request_stop()
    coord.join(threads)
    sess.close()
