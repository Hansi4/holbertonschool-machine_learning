#!/usr/bin/env python3
""" Initialize t-SNE """
import numpy as np


def P_init(X, perplexity):
    """ A python function that initializes all variables
    required to calculate the P affinities in t-SNE """

    n, d = X.shape

    # Compute the pairwise squared Euclidean distances (D)
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Set the diagonal to 0s
    np.fill_diagonal(D, 0)

    # Initialize P, betas, and H
    P = np.zeros((n, n))  # Affinity matrix initialized to 0
    betas = np.ones((n, 1))  # Beta values initialized to 1

    # Shannon entropy for given perplexity
    H = np.log2(perplexity)

    return D, P, betas, H
