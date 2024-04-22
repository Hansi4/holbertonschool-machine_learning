#!/usr/bin/env python3
""" Create Confusion """


import numpy as np


def create_confusion_matrix(labels, logits):
    """ A python function that creates a confusion matrix """
    m = labels.shape[0]
    classes = labels.shape[1]

    conf_matrix = np.zeros((classes, classes))

    for i in range(m):
        true_class = np.argmax(labels[i])
        predicted_class = np.argmax(logits[i])

        conf_matrix[true_class, predicted_class] += 1

    return conf_matrix
