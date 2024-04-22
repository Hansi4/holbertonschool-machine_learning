#!/usr/bin/env python3
""" Sensitivity """


import numpy as np


def sensitivity(confusion):
    """ A python function that calculates the
    sensitivity for each class in a confusion matrix """
    classes = confusion.shape[0]
    sensitivity_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        total_positives = np.sum(confusion[i, :])

        sensitivity_matrix[i] = true_positive / total_positives

    return sensitivity_matrix
