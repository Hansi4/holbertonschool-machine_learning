#!/usr/bin/env python3
""" that calculates the intersection of obtaining
this data with the various hypothetical probabilities """

import numpy as np


def intersection(x, n, P, Pr):
    """ a 1D numpy.ndarray containing the intersection
    of obtaining x and n with each probability in P, respectively """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or int(x) < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if int(x) > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
