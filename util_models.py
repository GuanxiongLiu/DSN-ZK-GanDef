"""
A pure TensorFlow implementation of neural network model.
This can be used directly as utils for neural network model with cleverhans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model

from util_layers import *



class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape, name):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        self.name = name
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            self.layer_names.append(layer.name)

    def fprop(self, x, set_ref=False):
        with tf.variable_scope(self.name):
            states = []
            for layer in self.layers:
                if set_ref:
                    layer.ref = x
                x = layer.fprop(x)
                assert x is not None
                states.append(x)
            states = dict(zip(self.get_layer_names(), states))
            return states

    def get_params(self):
        out = []
        for layer in self.layers:
            for param in layer.get_params(self.name):
                if param not in out:
                    out.append(param)
        return out

    
    

class GAN(Model):
    """
    An example of a bare bones generative adversarial net (GAN) class.
    """

    def __init__(self, clf_layers, dic_layers, input_shape, name):
        super(GAN, self).__init__()

        self.layer_names = []
        self.layers = clf_layers + dic_layers
        self.input_shape = input_shape
        self.name = name
        if isinstance(clf_layers[-1], Softmax):
            clf_layers[-1].name = 'clf_probs'
            clf_layers[-2].name = 'clf_logits'
        else:
            clf_layers[-1].name = 'clf_logits'
        if isinstance(dic_layers[-1], Softmax):
            dic_layers[-1].name = 'dic_probs'
            dic_layers[-2].name = 'dic_logits'
        else:
            dic_layers[-1].name = 'dic_logits'
        for i, layer in enumerate(self.layers):
            self.layer_names.append(layer.name)
            
    def __call__(self, *args, **kwargs):
        return self.get_gan_prob(*args, **kwargs)

    def fprop(self, x, set_ref=False):
        with tf.variable_scope(self.name):
            states = []
            for layer in self.layers:
                if set_ref:
                    layer.ref = x
                x = layer.fprop(x)
                assert x is not None
                states.append(x)
            states = dict(zip(self.get_layer_names(), states))
            return states

    def get_params(self):
        out = []
        for layer in self.layers:
            for param in layer.get_params(self.name):
                if param not in out:
                    out.append(param)
        return out
    
    def get_logits(self, x):
        return self.get_layer(x, 'clf_logits')
    
    def get_gan_params(self):
        clf, dic = [], []
        for layer in self.layers:
            for param in layer.get_params(self.name):
                if layer.name[:3] == 'clf' and param not in clf:
                    clf.append(param)
                if layer.name[:3] == 'dic' and param not in dic:
                    dic.append(param)
        return clf, dic
    
    def get_gan_prob(self, x):
        output = self.fprop(x)
        # classification prob
        try:
            clf_prob = output['clf_probs']
        except KeyError:
            pass
        except NotImplementedError:
            pass
        # discrimination prob
        try:
            dic_prob = output['dic_probs']
            return clf_prob, dic_prob
        except KeyError:
            pass
        except NotImplementedError:
            pass
        # prob does not exist
        clf_prob = tf.nn.softmax(output['clf_logits'])
        dic_prob = tf.nn.sigmoid(output['dic_logits'])
        
        return clf_prob, dic_prob

    
    

