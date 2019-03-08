"""
This is a tensorflow implementation of pure 
adversarial training PGD which is presented 
in its original paper from the link below.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl
from cleverhans.utils import AccuracyReport, set_log_level

import sys
sys.path.append('../')
from util_layers import *
from util_models import *
from util_train import *
from util_eval import *
from util_loaders import *
from util_augment import *

import os

FLAGS = flags.FLAGS


def make_zero_knowledge_gandef_model(name, nb_classes=10, input_shape=(None, 28, 28, 1)):
    clf_layers = [Conv2D(output_channels=32, 
                         kernel_shape=(5, 5), 
                         strides=(1, 1), 
                         padding="SAME", 
                         name="clf_conv2d_1",  
                         kernel_initializer=None),
                  MaxPool2D(window_size=(2,2), 
                            strides=(2,2), 
                            name="clf_maxpool2d_1"),
                  ReLU(name="clf_act_1"),
                  Conv2D(output_channels=64, 
                         kernel_shape=(5, 5), 
                         strides=(1, 1), 
                         padding="SAME", 
                         name="clf_conv2d_2",  
                         kernel_initializer=None),
                  MaxPool2D(window_size=(2,2), 
                            strides=(2,2), 
                            name="clf_maxpool2d_2"),
                  ReLU(name="clf_act_2"),
                  Flatten(name="clf_flat_1"),
                  Dense(num_hid=1024, 
                        name="clf_dense_1", 
                        kernel_initializer=None),
                  ReLU(name="clf_act_3"),
                  Dense(num_hid=nb_classes, 
                        name="clf_dense_2", 
                        kernel_initializer=None)]
    dic_layers = [Dense(num_hid=32, 
                        name="dic_dense_1", 
                        kernel_initializer=None),
                  ReLU(name="dic_act_1"),
                  Dense(num_hid=64, 
                        name="dic_dense_2", 
                        kernel_initializer=None),
                  ReLU(name="dic_act_2"),
                  Dense(num_hid=1, 
                        name="dic_dense_3", 
                        kernel_initializer=None)]

    model = GAN(clf_layers, dic_layers, input_shape, name)
    return model


def train_zero_knowledge_gandef_model(train_start=0, train_end=60000, test_start=0, 
                                      test_end=10000, smoke_test=True, save=False, testing=False, 
                                      backprop_through_attack=False, num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param train_batch_size: size of training batches
    :param test_batch_size: size of testing batches
    :param learning_rate: learning rate for training
    :param save: if true, the final model will be saved
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    if smoke_test:
        X_train, Y_train, X_test, Y_test = X_train[:256], Y_train[:256], X_test[:256], Y_test[:256]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y_soft = tf.placeholder(tf.float32, shape=(None, 10))

    # Train an MNIST model
    train_params = {
        'nb_epochs': 80,
        'batch_size': 128,
        'trade_off': 2,
        'inner_epochs': 1
    }
    learning_rate = 1e-4
    rng = np.random.RandomState([2017, 8, 30])

    # Adversarial training
    print("Start adversarial training")
    zero_knowledge_gandef_model = make_zero_knowledge_gandef_model(name="model_zero_knowledge_gandef")
    aug_x = gaussian_augment(x, std=1)
    preds_clean = zero_knowledge_gandef_model(x)
    preds_aug = zero_knowledge_gandef_model(aug_x)

    def cross_entropy(truth, preds, mean=True):
        # Get the logits operator
        op = preds.op
        if op.type == "Softmax":
            logits, = op.inputs
        else:
            logits = preds

        # Calculate cross entropy loss
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth)

        # Take average loss and return
        if mean:
            out = tf.reduce_mean(out)
        return out
    
    def sigmoid_entropy(truth, preds, mean=True):
        # Get the logits operator
        op = preds.op
        if op.type == "Softmax":
            logits, = op.inputs
        else:
            logits = preds

        # Calculate cross entropy loss
        out = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=truth)

        # Take average loss and return
        if mean:
            out = tf.reduce_mean(out)
        return out

    # Perform and evaluate adversarial training
    clf_opt = tf.train.AdamOptimizer(learning_rate)
    dic_opt = tf.train.AdamOptimizer(learning_rate*10)
    gan_train_v2(sess, x, y_soft, preds_clean, X_train, Y_train, 
                 loss_func=[cross_entropy, sigmoid_entropy], optimizer=[clf_opt, dic_opt], 
                 predictions_adv=preds_aug, evaluate=None, args=train_params, rng=rng, 
                 var_list=zero_knowledge_gandef_model.get_gan_params())
    
    # Evaluate the accuracy of the MNIST model on Clean examples
    preds_clean = zero_knowledge_gandef_model(x)
    eval_params = {
        'batch_size': 128,
        'use_dic': False,
        'is_clean': True,
        'reject_threshold': 0.5
    }
    clean_acc = confident_model_eval(sess, x, y_soft, preds_clean, X_test, Y_test, args=eval_params)
    print('Test accuracy on Clean test examples: %0.4f\n' % clean_acc)
    report.adv_train_clean_eval = clean_acc
    
    # Evaluate the accuracy of the MNIST model on FGSM examples
    fgsm_params = {
        'eps': 0.6,
        'clip_min': -1.,
        'clip_max': 1.
    }
    fgsm_att = FastGradientMethod(zero_knowledge_gandef_model, sess=sess)
    fgsm_adv = fgsm_att.generate(x, **fgsm_params)
    preds_fgsm_adv = zero_knowledge_gandef_model(fgsm_adv)
    eval_params = {
        'batch_size': 128,
        'use_dic': False,
        'is_clean': False,
        'reject_threshold': 0.5
    }
    fgsm_acc = confident_model_eval(sess, x, y_soft, preds_fgsm_adv, X_test, Y_test, args=eval_params)
    print('Test accuracy on FGSM test examples: %0.4f\n' % fgsm_acc)
    report.adv_train_adv_eval = fgsm_acc
    
    # Evaluate the accuracy of the MNIST model on BIM examples
    bim_params = {
        'eps': 0.6,
        'eps_iter': 0.1,
        'clip_min': -1.,
        'clip_max': 1.
    }
    bim_att = BasicIterativeMethod(zero_knowledge_gandef_model, sess=sess)
    bim_adv = bim_att.generate(x, **bim_params)
    preds_bim_adv = zero_knowledge_gandef_model(bim_adv)
    eval_params = {
        'batch_size': 128,
        'use_dic': False,
        'is_clean': False,
        'reject_threshold': 0.5
    }
    bim_acc = confident_model_eval(sess, x, y_soft, preds_bim_adv, X_test, Y_test, args=eval_params)
    print('Test accuracy on BIM test examples: %0.4f\n' % bim_acc)
    report.adv_train_adv_eval = bim_acc
    
    # Evaluate the accuracy of the MNIST model on PGD examples
    pgd_params = {
        'eps': 0.6,
        'eps_iter': 0.02,
        'nb_iter': 40,
        'clip_min': 0.,
        'clip_max': 1., 
        'rand_init': True
    }
    pgd_att = MadryEtAl(zero_knowledge_gandef_model, sess=sess)
    pgd_adv = pgd_att.generate(x, **bim_params)
    preds_pgd_adv = zero_knowledge_gandef_model(pgd_adv)
    eval_params = {
        'batch_size': 128,
        'use_dic': False,
        'is_clean': False,
        'reject_threshold': 0.5
    }
    pgd_acc = confident_model_eval(sess, x, y_soft, preds_pgd_adv, X_test, Y_test, args=eval_params)
    print('Test accuracy on PGD test examples: %0.4f\n' % pgd_acc)
    report.adv_train_adv_eval = pgd_acc
    
    # Save model
    if save:
        model_path = "models/zero_knowledge_gandef"
        vars_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                         scope='model_zero_knowledge_gandef*')
        assert len(vars_to_save) > 0
        saver = tf.train.Saver(var_list=vars_to_save)
        saver.save(sess, model_path)
        print('Model saved\n')
    else:
        print('Model not saved\n')


def main(argv=None):
    train_zero_knowledge_gandef_model(smoke_test=FLAGS.smoke_test, 
                                      save=FLAGS.save, 
                                      backprop_through_attack=FLAGS.backprop_through_attack)


if __name__ == '__main__':
    flags.DEFINE_bool('smoke_test', False, 'Smoke test')
    flags.DEFINE_bool('save', True, 'Save model')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
