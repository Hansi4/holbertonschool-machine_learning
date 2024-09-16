#!/usr/bin/env python3
""" Principal Component Analysis """
import numpy as np


def pca(X, var=0.95):
    """ A python function that performs PCA on a dataset """

    # computing the covariance matrix of X,
    # rowvar=False ensures that columns represent features (dimensions).
    covariance_matrix = np.cov(X, rowvar=False)
    
    # Performing eigenvalue decomposition,
    # np.linalg.eigh is used for eigenvalue decomposition of symmetric matrices like the covariance matrix. 
    eigen_value, eigen_vector = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues in desc order
    sorted_indices = np.argsort(eigen_value)[: : -1]
    sorted_eigen_value = eigen_value[sorted_indices]
    sorted_eigen_vector = eigen_vector[:, sorted_indices]
    
    # compute the cumulative value of the sorted eigenvalues
    cumulative_variance = np.cumsum(sorted_eigen_value) / np.sum(sorted_eigen_value)
    
    # select minimum number of components to maintain the desired variance
    n_componets = np.argmax(cumulative_variance >= var) + 2
    n_componets = abs(n_componets)
    
    # Make the w matrix with the correct values
    W = sorted_eigen_vector[:, :n_componets]
    
    return W
