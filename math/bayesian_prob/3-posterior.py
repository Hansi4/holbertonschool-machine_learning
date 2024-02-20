#!/usr/bin/env python3
""" that calculates the posterior probability for the various
hypothetical probabilities of developing severe side effects given the data """

import numpy as np


def posterior(x, n, P, Pr):
    """ the posterior probability of each
    probability in P given x and n, respectively """
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
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(x) * factorial(n-x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal = np.sum(intersection)

    posterior = intersection / marginal
    return posterior
