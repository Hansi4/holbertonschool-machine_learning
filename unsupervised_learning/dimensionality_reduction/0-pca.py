#!/usr/bin/env python3
""" Principal Component Analysis """
import numpy as np


def pca(X, var=0.95):
    """ A python function that performs PCA on a dataset """

    U, S, V = np.linalg.svd(X)

    ratios = list(x / np.sum(S) for x in S)

    variance = np.cumsum(ratios)

    nd = np.argwhere(variance >= var)[0, 0]

    W = V.T[:, :(nd + 1)]

    return (W)
