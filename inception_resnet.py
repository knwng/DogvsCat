import tensorflow as tf

from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch. height, width, 3]')
    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
        _, endpoints = inception_resnet_v2(image, num_classes=2, is_training=is_training)
        
