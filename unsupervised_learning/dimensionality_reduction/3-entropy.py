#!/usr/bin/env python3
""" Entropy """
import numpy as np


def HP(Di, beta):
    """ A python function that calculates the Shannon
    entropy and P affinities relative to a data point """

    # Compute the P affinities
    Pi = np.exp(-Di * beta)  # Apply Gaussian distribution
    sum_Pi = np.sum(Pi)  # Sum of all affinities
    Pi = Pi / sum_Pi  # Normalize to get probabilities

    # Compute Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi))  # Shannon entropy formula

    return Hi, Pi
