#!/usr/bin/env python3
""" One Hot Encoding """


import numpy as np


def one_hot_encode(Y, classes):
    """ Method that converts a numerical label vector into a
        one-hot vector """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    encoded_array = np.zeros((classes, Y.size), dtype=float)

    encoded_array[Y, np.arange(Y.size)] = 1

    return encoded_array
