#!/usr/bin/env python3
""" that calculates the likelihood of obtaining this data given various 
hypothetical probabilities of developing severe side effects """


def likelihood(x, n, P):
    """ returns a 1D numpy.ndarray containing the likelihood of obtaining 
    the data, x and n, for each probability in P, respectively """
    