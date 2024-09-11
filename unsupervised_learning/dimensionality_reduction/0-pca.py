#!/usr/bin/env python3
""" Principal Component Analysis """
import numpy as np


def pca(X, var=0.95):
    """ A python function that performs PCA on a dataset """

    # Step 1: Compute the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Step 2: Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 3: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4: Compute the cumulative variance
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

    # Step 5: Select the number of components to maintain the desired variance
    num_components = np.searchsorted(cumulative_variance, var) + 1

    # Step 6: Return the weight matrix W
    W = sorted_eigenvectors[:, :num_components]

    return W
