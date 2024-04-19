#!/usr/bin/env python3
""" Batch Normalization """


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ A python function that normalizes an unactivated output
    of a neural network using batch normalization """
    # compute mean and variance
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    std_dev = np.sqrt(variance + epsilon)

    # normalize : - mean / root square variance
    Z_norm = (Z - mean) / std_dev

    # scale by gama and shifted by beta
    scaled = gamma * Z_norm + beta

    return scaled
