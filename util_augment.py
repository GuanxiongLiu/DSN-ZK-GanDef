"""
A pure TensorFlow implementation of different augmentation functions.
This can be used directly as utils for neural network model with cleverhans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf




def cifar10_augment(tensor_in):
    row, col, channel = 32, 32, 3
    # padding
    padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,
                                                                          row + 4,
                                                                          col + 4),
                       tensor_in)
    # crop
    cropped = tf.map_fn(lambda img: tf.random_crop(img, [row,
                                                         col,
                                                         channel]),
                        padded)
    # flip
    flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
    return flipped



def gaussian_augment(tensor_in, mean=0.0, std=1.0):
    noise = tf.random_normal(tf.shape(tensor_in), mean=mean, stddev=std)
    augmented = tf.clip_by_value(tf.add(tensor_in, noise), -1, 1)
    return augmented
