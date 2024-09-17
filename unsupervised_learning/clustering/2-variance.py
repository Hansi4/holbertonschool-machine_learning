#!/usr/bin/env python3
""" Variance """
import numpy as np


def variance(X, C):
    """ A python function that calculates
    the total intra-cluster variance for a data set """

    n, d = X.shape
    centroids_extended = C[:, np.newaxis]
    distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))

    min_distances = np.min(distances, axis=0)
    variances = np.sum(min_distances ** 2)

    return variance
