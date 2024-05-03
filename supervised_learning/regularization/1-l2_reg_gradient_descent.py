#!/usr/bin/env python3
""" Gradient Descent with L2 Regularization """


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ A python function that updates the weights and biases
    of a neural network using gradient descent with L2 regularization """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        L2_regularization = lambtha / m * weights['W' + str(layer)]

        # activation function
        A_prev = cache['A' + str(layer - 1)]

        # gradient calculation
        dW = np.matmul(dZ, A_prev.T) / m + L2_regularization
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights['W' + str(layer)].T, dZ)

        # differentiate between layers
        if layer != 1:
            dZ = dA * (1 - A_prev ** 2)
        else:
            dZ = dA

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
