#!/usr/bin/env python3
""" Precision """


import numpy as np


def precision(confusion):
    """ A python function that calculates the
    precision for each class in a confusion matrix """
    classes = confusion.shape[0]
    precision_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        false_positive = np.sum(confusion[:, i]) - true_positive

        precision_matrix[i] = true_positive / (true_positive + false_positive)

    return precision_matrix
