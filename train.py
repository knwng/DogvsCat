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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

slim = tf.contrib.slim

ap = argparse.ArgumentParser()
ap.add_argument('--resume', action='store_true', default=False)
ap.add_argument('--train_dataset', type=common.readable_directory, required=True)
ap.add_argument('--val_dataset', type=common.readable_directory, required=True)
ap.add_argument('--train_dir', type=common.readable_directory, required=True)
ap.add_argument('--learning_rate', default=1e-4, type=common.positive_float)
ap.add_argument('--epoch', default=5000, type=common.positive_int)
ap.add_argument('--batch_size', default=32, type=common.positive_int)
ap.add_argument('--image_size', default=224, type=common.positive_int)
ap.add_argument('--pretrained_model', default='./pretrained_model/inception_resnet_v2.ckpt', type=common.readable_directory)

args = ap.parse_args()

if not os.path.isdir(args.train_dir):
    os.system('mkdir -p {}'.format(args.train_dir))

_RGB_MEAN = [123.68, 116.78, 103.94]

time_identifier = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.train_dir, '{}_{}.log'.format('log', time_identifier)))
fh.setLevel(logging.INFO)
log.addHandler(fh)

def get_batch(content, is_train):
    dataset = tf.data.TextLineDataset(content)
    dataset = dataset.map(decode_line).shuffle(buffer_size=1000).batch(args.batch_size)
    if is_train:
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat(1)
    return dataset.make_initializable_iterator()

def decode_line(line):
    dir_, label = tf.decode_csv(records=line, record_defaults=[["string"], ["string"]], field_delim=" ")
    label = tf.one_hot(tf.string_to_number(label, tf.int32), depth=2, dtype=tf.float32)
    imagecontent = tf.read_file(dir_)
    image = tf.image.decode_png(imagecontent, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [args.image_size, args.image_size])
    image = (image - _RGB_MEAN) / 128.
    return image, label

train_iter = get_batch(content=[args.train_dataset], is_train=True)
val_iter = get_batch(content=[args.val_dataset], is_train=False)

def validation(sess, val_iter):
    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
        image_op = tf.get_default_graph().get_tensor_by_name('image_input:0')
        label_op = tf.get_default_graph().get_tensor_by_name('label_input:0')
        acc, op_acc = tf.get_default_graph().get_tensor_by_name('val_acc/value:0'), tf.get_default_graph().get_tensor_by_name('val_acc/update_op:0')
        precision, op_precision = tf.get_default_graph().get_tensor_by_name('val_precision/value:0'), tf.get_default_graph().get_tensor_by_name('val_precision/update_op:0')
        recall, op_recall = tf.get_default_graph().get_tensor_by_name('val_recall/value:0'), tf.get_default_graph().get_tensor_by_name('val_recall/update_op:0')
    image, label = val_iter.get_next()
    sess.run(val_iter.initializer)
    try:
        while True:
            _images, _labels = sess.run([image, label])
            sess.run([op_acc, op_precision, op_recall], feed_dict={image_op: _images, label_op: _labels})
    except tf.errors.OutOfRangeError:
        pass
    finally:
        _acc, _precision, _recall = sess.run([acc, precision, recall])
    return _acc, _precision, _recall

with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
    images = tf.placeholder(tf.float32, (None, args.image_size, args.image_size, 3), name='image_input')
    labels = tf.placeholder(tf.float32, (None, 2), name='label_input')
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
tf.summary.scalar('learning_rate', learning_rate)
loss = tf.losses.log_loss(labels, predictions, weights=10)
tf.summary.scalar('loss', loss)
metric_acc, op_acc = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1), name='train_acc')
eval_metric_acc, eval_op_acc = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1), name='val_acc')
eval_metric_p, eval_op_p = tf.metrics.precision(tf.argmax(labels, 1), tf.argmax(predictions, 1), name='val_precision')
eval_metric_r, eval_op_r = tf.metrics.recall(tf.argmax(labels, 1), tf.argmax(predictions, 1), name='val_recall')
tf.summary.scalar('train_acc', metric_acc)
tf.summary.scalar('val_acc', eval_metric_acc)
tf.summary.scalar('val_precision', eval_metric_p)
tf.summary.scalar('val_recall', eval_metric_r)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
ckpt_saver = tf.train.Saver(max_to_keep=0)

train_record_fn = os.path.join(args.train_dir, 'train-{}.csv'.format(time_identifier))
val_record_fn = os.path.join(args.train_dir, 'val-{}.csv'.format(time_identifier))

with tf.Session(config=config) as sess, open(train_record_fn, 'w') as ftrain, open(val_record_fn, 'w') as fval:
    if args.resume:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        last_checkpoint = tf.train.latest_checkpoint(args.train_dir)
        saver.restore(sess, last_checkpoint)
        tf.logging.info('Resume training from [{}]'.format(last_checkpoint))
    else:
        saver = tf.train.Saver(model_variables)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tf.logging.info('Load pretrained model from [{}]'.format(args.pretrained_model))
        saver.restore(sess, args.pretrained_model)
    train_writer = csv.writer(ftrain)
    train_writer.writerow(['global_step', 'loss', 'accuracy'])
    val_writer = csv.writer(fval)
    val_writer.writerow(['global_step', 'accuracy', 'precision', 'recall'])

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)
    train_image, train_label = train_iter.get_next()

    try:
        for _epoch in range(int(args.epoch)):
            sess.run(train_iter.initializer)
            while True:
                try:
                    start_time = time.time()
                    _images, _labels = sess.run([train_image, train_label])
                    _, _lr, _summary, _step, _acc, _loss = sess.run([train_op, learning_rate, merged_summary, global_step, op_acc, loss], feed_dict={images: _images, labels: _labels})
                    elapsed_time = time.time() - start_time
                    tf.logging.info("Time: {} | Learning Rate: {} | Epoch: {} | Global_Step: {} | Loss: {} | Accuracy: {}".format(elapsed_time, _lr, _epoch, _step, _loss, _acc))
                    train_writer.writerow([_step, _loss, _acc])
                    ftrain.flush()
                    if _step % 1000 == 0:
                        ckpt = ckpt_saver.save(sess, os.path.join(args.train_dir, 'model'), global_step=_step)
                        tf.logging.info('Save ckpt-{} in {}'.format(_step, ckpt))
                    if _step % 500 == 0:
                        _eval_acc, _eval_p, _eval_r = validation(sess, val_iter)
                        val_writer.writerow([_step, _eval_acc, _eval_p, _eval_r])
                        fval.flush()
                        tf.logging.info('*'*80)
                        tf.logging.info("Validation | Global_Step: {} | Accuracy: {} | Precision: {} | Recall: {}".format(_step, _eval_acc, _eval_p, _eval_r))
                        tf.logging.info('*'*80)
                except tf.errors.OutOfRangeError:
                    break
    except tf.errors.OutOfRangeError:
        tf.logging.info('Finished')
    finally:
        ckpt = ckpt_saver.save(sess, os.path.join(args.train_dir, 'model'), global_step=tf.train.get_global_step())
        tf.logging.info('Save ckpt in: {}'.format(ckpt))
    sess.close()

