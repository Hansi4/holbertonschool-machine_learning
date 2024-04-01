#!/usr/bin/env python3
""" One hot encoding """


import numpy as np


def one_hot_encode(Y, classes):
    """ one hot encode """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None

    encoded_array = np.zeros((classes, Y.size), dtype=float)
    encoded_array[Y, np.arange(Y.size)] = 1
    return encoded_array

