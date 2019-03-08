"""
A pure TensorFlow implementation of different training functions.
This can be used directly as utils for neural network model with cleverhans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import math
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
import time
import warnings

from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")



def gan_train_v2(sess, x, y, predictions, X_train, Y_train,
                 loss_func=None, optimizer=None, predictions_adv=None, init_all=True,
                 evaluate=None, feed=None, args=None, rng=None, var_list=None, ):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions [class_pred, source_pred]
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param trade_off: balance trade off between classifier and discriminator loss
    :param loss_func: list of loss functions [clf_loss, dic_loss]
    :param optimizer: tensorflow optimizer
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training [adv_class_pred, adv_source_pred]
    :param init_all: (boolean) If set to true, all TF variables in the session
                     are (re)initialized, otherwise only previously
                     uninitialized variables are initialized before training.
    :param evaluate: function that is run after each training iteration
                     (typically to display the test/validation accuracy).
    :param feed: An optional dictionary that is appended to the feeding
                 dictionary before the session runs. Can be used to feed
                 the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'train_dir'
                 and 'filename'
    :param rng: Instance of numpy.random.RandomState
    :param var_list: Optional list of parameters to train.
    :return: True if model trained
    """
    args = _ArgsWrapper(args or {})
    
    # Check that necessary inputs were given
    assert len(predictions) == 2, "Number of prediction inputs was not match"
    assert len(predictions_adv) == 2, "Number of adversarial prediction inputs was not match"
    assert len(var_list) == 2, "Number of variable list was not match"

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"
    assert args.trade_off, "Balance parameter was not given in args dict"
    assert args.inner_epochs, "Number of inner epochs was not given in args dict"
    
    # Check that necessary operators were given
    assert len(loss_func) == 2, "Number of loss function was not match"
    assert len(optimizer) == 2, "Number of optimizer was not match"

    if rng is None:
        rng = np.random.RandomState()
        
    # Define discriminator loss
    adv_source_loss = loss_func[1](tf.ones(shape=[tf.shape(y)[0], 1]), predictions_adv[1])
    dic_loss = (loss_func[1](tf.zeros(shape=[tf.shape(y)[0], 1]), predictions[1]) + adv_source_loss) / 2

    # Define classifier loss
    class_loss = loss_func[0](y, predictions[0])
    pre_loss = (class_loss + loss_func[0](y, predictions_adv[0])) / 2
    clf_loss = pre_loss - args.trade_off * adv_source_loss
    
    # Add weight decay
    if args.weight_decay is not None:
        weights = []
        for var in tf.trainable_variables():
            if var.op.name.find('clf') > 0 and var.op.name.find('kernel') > 0:
                weights.append(tf.nn.l2_loss(var))
        weight_loss = args.weight_decay * tf.add_n(weights)
        pre_loss += weight_loss
        clf_loss += weight_loss
    
    # Define training operation
    if args.global_step is not None:
        pre_step = optimizer[0].minimize(pre_loss, var_list=var_list[0], global_step=args.global_step)
        clf_step = optimizer[0].minimize(clf_loss, var_list=var_list[0], global_step=args.global_step)
    else:
        pre_step = optimizer[0].minimize(pre_loss, var_list=var_list[0])
        clf_step = optimizer[0].minimize(clf_loss, var_list=var_list[0])
    dic_step = optimizer[1].minimize(dic_loss, var_list=var_list[1])

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            if init_all:
                tf.global_variables_initializer().run()
            else:
                initialize_uninitialized_global_variables(sess)
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "CleverHans may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        for epoch in xrange(args.nb_epochs):
            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)

            prev = time.time()
            
            if epoch < args.pretrain_epochs:
                # Pre-train Classifier
                _logger.info("Pre-train Epoch")
                for batch in range(nb_batches):
                    # Train Classifier
                    # Compute batch start and end indices
                    start, end = batch_indices(batch, len(X_train), args.batch_size)
                    # Perform one training step
                    feed_dict = {x: X_train[index_shuf[start:end]],
                                 y: Y_train[index_shuf[start:end]]}
                    if feed is not None:
                        feed_dict.update(feed)
                    pre_step.run(feed_dict=feed_dict)
            else:
                # GAN Training
                _logger.info("GAN-train Epoch")
                for batch in range(nb_batches):                
                    # Train Discriminator
                    inner_batches = np.random.choice(nb_batches, args.inner_epochs)
                    for inner_batch in inner_batches:
                        # Compute batch start and end indices
                        inner_start, inner_end = batch_indices(inner_batch, len(X_train), args.batch_size)
                        # Perform one training step
                        feed_dict = {x: X_train[index_shuf[inner_start:inner_end]],
                                     y: Y_train[index_shuf[inner_start:inner_end]]}
                        if feed is not None:
                            feed_dict.update(feed)
                        dic_step.run(feed_dict=feed_dict)
                    # Train Classifier
                    # Compute batch start and end indices
                    start, end = batch_indices(batch, len(X_train), args.batch_size)
                    # Perform one training step
                    feed_dict = {x: X_train[index_shuf[start:end]],
                                 y: Y_train[index_shuf[start:end]]}
                    if feed is not None:
                        feed_dict.update(feed)
                    '''
                    clf_step.run(feed_dict=feed_dict)
                    '''
                    _, cl, dl = sess.run(fetches=[clf_step, pre_loss, dic_loss], 
                                         feed_dict=feed_dict)
                    
            # check loss
            _logger.info("Epoch %d - Classifier Loss %4f - Discriminator Loss %4f " % (epoch, cl, dl))
            
            # Check that all examples were used
            assert end >= len(X_train)  
            cur = time.time()
            _logger.info("Epoch " + str(epoch) + " took " +
                         str(cur - prev) + " seconds")
            
            if evaluate is not None:
                evaluate()

        _logger.info("Completed model training.")

    return True



