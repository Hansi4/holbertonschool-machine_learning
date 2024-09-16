#!/usr/bin/env python3
""" Principal Component Analysis v2 """
import numpy as np


def pca(X, ndim):
    """ A python function that performs PCA on a dataset """

    # Step 1: Center the data (subtract the mean of each feature)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, V = np.linalg.svd(X_centered)
    W = V.T[:, :ndim]
    T = np.matmul(X_centered, W)

    return T
