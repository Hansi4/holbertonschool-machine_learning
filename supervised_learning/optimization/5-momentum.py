#!/usr/bin/env python3
""" Momentum """


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ A python function that updates a variable
    using the gradient descent with
    momentum optimization algorithm """

    # formula for momentum
    dW = beta1 * v + (1 - beta1) * grad

    # update variable
    var_new = var - dW * alpha

    return var_new, dW
