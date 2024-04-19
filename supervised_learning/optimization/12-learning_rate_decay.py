#!/usr/bin/env python3
""" Learning Rate Decay Upgraded """


import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ A python function that creates a learning rate
    decay operation in tensorflow using inverse time decay """
    # set train exponential decay in tf
    # use staircase=True to occur in a stepwise fashion
    learning_rate = tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        global_step=global_step,
        staircase=True)

    return learning_rate
