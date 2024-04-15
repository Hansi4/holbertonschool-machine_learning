#!/usr/bin/env python3
""" Shuffle Data """


import numpy as np


def shuffle_data(X, Y):
    """ A python function that shuffles
    the data points in two matrices the same way """
    m = X.shape[0]
    permutted_index = np.random.permutation(m)

    X_shuffled = X[permutted_index]
    Y_shuffled = Y[permutted_index]

    return X_shuffled, Y_shuffled
