#!/usr/bin/env python3
""" Batch Normalization Upgraded """


import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """ A python function that creates a batch normalization
    layer for a neural network in tensorflow """
    # set initialization to He et. al
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # create layer Dense with parameters
    new_layer = tf.layers.Dense(n,
                                activation=None,
                                kernel_initializer=initializer,
                                name="layer")

    # apply layer to input
    x = new_layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    # beta and gamma : two trainable parameters
    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')

    epsilon = 1e-8

    # apply batch normalization
    x_norm = tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(x_norm)
