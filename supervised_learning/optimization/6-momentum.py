#!/usr/bin/env python3
""" Momentum upgraded """


import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ A python function that creates the training operation
    for a neural network in tensorflow using the gradient
    descent with momentum optimization algorithm """

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1
    )
    train_op = optimizer.minimize(loss)

    return train_op
