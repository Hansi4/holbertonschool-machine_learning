#!/usr/bin/env python3
""" Entropy """
import numpy as np


def HP(Di, beta):
    """ A python function that calculates the Shannon
    entropy and P affinities relative to a data point """

    # Ensure beta is a scalar
    beta = beta.item()

    # Calculate P affinities using the Gaussian distribution
    P = np.exp(-Di ** 2 / (2 * beta ** 2))

    # Normalize P affinities to sum to 1
    Pi = P / np.sum(P)

    # Calculate Shannon entropy
    Hi = -np.sum(Pi * np.log(Pi + 1e-10))  # Adding a small value to avoid log(0)

    return Hi, Pi
