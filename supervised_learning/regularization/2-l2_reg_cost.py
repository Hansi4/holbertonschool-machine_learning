#!/usr/bin/env python3
""" L2 Regularization Cost """


import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ A python function that calculates the cost
    of a neural network with L2 regularization """

    reg_term = 0
    for i in range(1, L+1):
        W_i = weights['W' + str(i)]
        reg_term += np.sum(np.square(W_i))

    cost_L2 = cost + (lambtha / (2*m)) * reg_term

    return cost_L2
