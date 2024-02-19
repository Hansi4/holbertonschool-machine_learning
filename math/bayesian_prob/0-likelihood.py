#!/usr/bin/env python3
""" that calculates the likelihood of 
obtaining this data given various
hypothetical probabilities of developing severe side effects """

import numpy as np


def likelihood(x, n, P):
    """ returns a 1D numpy.ndarray containing the likelihood of obtaining 
    the data, x and n, for each probability in P, respectively """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or int(x) < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if int(x) > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
