#!/usr/bin/env python3
""" F1 score """


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

def f1_score(confusion):
    """ A python function that calculates the
    F1 score of a confusion matrix """
    classes = confusion.shape[0]
    f_one_matrix = np.zeros((classes,))

    prec_value = precision(confusion)
    rec_value = sensitivity(confusion)

    for i in range(classes):
        f_one_matrix[i] = 2 * prec_value[i] * rec_value[i] / (prec_value[i] + rec_value[i])

    return f_one_matrix
