#!/usr/bin/env python3
""" that calculates the likelihood of obtaining this data given various 
hypothetical probabilities of developing severe side effects """


def likelihood(x, n, P):
    """ returns a 1D numpy.ndarray containing the likelihood of obtaining 
    the data, x and n, for each probability in P, respectively """
    if n < 0:
        raise ValueError("n must be a positive integer")
    if x <= 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    else:
        if type(P) is not numpy.ndarray:
            raise TypeError("P must be a 1D numpy.ndarray")  