#!/usr/bin/env python3
""" Create a Layer with Dropout """


import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ A python function that creates
    a layer of a neural network using dropout """

    dropout_layer = tf.compat.v1.layers.Dropout(rate=keep_prob)

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    new_layer = (
        tf.layers.Dense(n,
                        activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=dropout_layer,
                        name="layer"))

    output = new_layer(prev)

    return output
