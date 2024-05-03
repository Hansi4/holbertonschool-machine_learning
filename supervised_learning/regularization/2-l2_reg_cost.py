#!/usr/bin/env python3
""" L2 Regularization Cost """


import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ A python function that calculates the cost
    of a neural network with L2 regularization """

    regularization_L2 = tf.compat.v1.losses.get_regularization_losses()

    cost_L2reg = cost + regularization_L2

    return cost_L2reg