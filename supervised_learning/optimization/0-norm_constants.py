#!/usr/bin/env python3
""" Normalization constants """


import numpy as np


def normalization_constants(X):
    """ A python function that calculates the normalization
    (standardization) constants of a matrix """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
