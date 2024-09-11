#!/usr/bin/env python3
""" Principal Component Analysis v2 """
import numpy as np


def pca(X, ndim):
    """ A python function that performs PCA on a dataset """

    # Step 1: Center the data (subtract the mean of each feature)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Calculate the covariance matrix of the centered data
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    # Step 5: Select the top `ndim` eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :ndim]

    # Step 6: Transform the data to the new subspace
    T = np.dot(X_centered, selected_eigenvectors)

    return T
