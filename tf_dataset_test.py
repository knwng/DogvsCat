#!/bin/env python
import tensorflow as tf
import numpy as np
import os

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["label"]

def decode_line(line):
    dir_, label = tf.decode_csv(records=line, record_defaults=[["string"], ["string"]], field_delim=" ")
    label = tf.one_hot(tf.string_to_number(label, tf.int32), depth=2, dtype=tf.float32)
    # print('Image Dir: {} | Label: {}'.format(dir_, label))
    imagecontent = tf.read_file(dir_)
    image = tf.image.decode_png(imagecontent, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [224, 224])
    image = (image - [123.68, 116.78, 103.94]) / 128.
    return image, label
    # return dir_, label

filenames = './dataset/dogsvscats/train.txt'
dataset = tf.data.TextLineDataset([filenames])
dataset = dataset.map(decode_line)
dataset = dataset.shuffle(buffer_size=10000).batch(128).repeat(1)
iterator = dataset.make_initializable_iterator()
# iterator = dataset.make_one_shot_iterator()
next_image, next_label = iterator.get_next()
x = tf.placeholder(dtype=tf.float32, name='x')
y = tf.placeholder(dtype=tf.float32, name='y')

# with tf.Session() as sess:
with tf.train.MonitoredTrainingSession() as sess:
    for _ in range(100):
        sess.run(iterator.initializer)
        while True:
            try:
                _image, _label = sess.run([next_image, next_label])
                image, label = sess.run([x, y], feed_dict={x:_image, y:_label})
                print('Image: {} | Label: {}'.format(image.shape, label.shape))
            except tf.errors.OutOfRangeError:
                print('Finish One Epoch')
                break


