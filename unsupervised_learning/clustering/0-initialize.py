#!/usr/bin/env python3
""" Initialize K-means """
import numpy as np


def initialize(X, k):
    """ A python function that initializes cluster centroids for K-means """

    if not isinstance(X, numpy.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    # Setting min and max values per col
    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # return multivariate uniform distribution
    return numpy.random.uniform(X_min, X_max, size=(k, d))
