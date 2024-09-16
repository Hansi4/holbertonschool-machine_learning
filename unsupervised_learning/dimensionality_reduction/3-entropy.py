#!/usr/bin/env python3
""" Entropy """
import numpy as np


def HP(Di, beta):
    """ A python function that calculates the Shannon
    entropy and P affinities relative to a data point """

    # Compute the P affinities using the Gaussian distribution
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi  # Normalize to get affinities

    # Compute the Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-10))  # Adding a small epsilon to avoid log(0)

    return Hi, Pi
