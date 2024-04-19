#!/usr/bin/env python3
""" Adam Upgraded """


import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ A python function that creates the training operation for a
    neural network in tensorflow using the Adam optimization algorithm """

    # set optimizer that implement Adam algo in tf
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    # train_op to minimize loss with this optimizer
    train_op = optimizer.minimize(loss)

    return train_op
