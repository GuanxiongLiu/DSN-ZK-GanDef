"""
A pure TensorFlow implementation of different functional layers.
This can be used directly as utils for neural network model with cleverhans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf



class Dense():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fprop(self, x):
        return tf.layers.dense(inputs=x, units=self.num_hid, kernel_initializer=self.kernel_initializer,
                               name=self.name, reuse=tf.AUTO_REUSE)

    def get_params(self, model_name):
        with tf.variable_scope(model_name+'/'+self.name, reuse=True):
            W = tf.get_variable('kernel')
            b = tf.get_variable('bias')
        return [W, b]


class Conv2D():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fprop(self, x):
        return tf.layers.conv2d(inputs=x, filters=self.output_channels, kernel_size=self.kernel_shape, 
                                kernel_initializer=self.kernel_initializer, strides=self.strides,
                                padding=self.padding, name=self.name, reuse=tf.AUTO_REUSE)

    def get_params(self, model_name):
        with tf.variable_scope(model_name+'/'+self.name, reuse=True):
            W = tf.get_variable('kernel')
            b = tf.get_variable('bias')
        return [W, b]


class ReLU():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fprop(self, x):
        return tf.nn.relu(features=x, name=self.name)

    def get_params(self, model_name):
        return []


class Softmax():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fprop(self, x):
        return tf.nn.softmax(logits=x, name=self.name)

    def get_params(self, model_name):
        return []


class Flatten():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fprop(self, x):
        return tf.layers.flatten(inputs=x, name=self.name)

    def get_params(self, model_name):
        return []
    
    
class MaxPool2D():
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def fprop(self, x):
        return tf.layers.max_pooling2d(inputs=x, pool_size=self.window_size, strides=self.strides, name=self.name)
    
    def get_params(self, model_name):
        return []

    
class Dropout():
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def fprop(self, x):
        return tf.layers.dropout(inputs=x, rate=self.rate, name=self.name)
    
    def get_params(self, model_name):
        return []
    
    
class GlobalAvgPool2D():
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def fprop(self, x):
        return tf.keras.layers.GlobalAveragePooling2D()(x)
    
    def get_params(self, model_name):
        return []
    
    
class BatchNormalization():
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def fprop(self, x):
        return tf.layers.batch_normalization(inputs=x, name=self.name, reuse=tf.AUTO_REUSE)
    
    def get_params(self, model_name):
        return []