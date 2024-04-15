#!/usr/bin/env python3
""" RMSProp """


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ A python function that updates a
    variable using the RMSProp optimization algorithm """

    squared_gradient = beta2 * s + (1 - beta2) * (grad ** 2)
    update_var = var - (alpha * grad) / (np.sqrt(squared_gradient) + epsilon)

    return update_var, squared_gradient
