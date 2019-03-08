"""
A pure TensorFlow implementation of different loading functions.
This can be used directly as utils for neural network model with cleverhans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf




def data_mnist(**kwargs):
    from cleverhans.utils_mnist import data_mnist as original_data_mnist
    X_train, Y_train, X_test, Y_test = original_data_mnist(**kwargs)
    return (X_train-0.5)*2, Y_train, (X_test-0.5)*2, Y_test


def data_fashion_mnist(**kwargs):
    from keras.datasets import fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = X_train.reshape(X_train.shape+(1,))
    X_test = X_test.reshape(X_test.shape+(1,))
    
    from keras.utils import np_utils
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    return (X_train-0.5)*2, Y_train, (X_test-0.5)*2, Y_test


def data_cifar10(train_start=0, train_end=50000, test_start=0, test_end=10000):
    """
    Load and preprocess CIFAR10 dataset
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)
    
    from keras.datasets import cifar10
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    
    from keras.utils import np_utils
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    return (X_train-0.5)*2, Y_train, (X_test-0.5)*2, Y_test


