#!/usr/bin/env python3
""" defines function that calculates the shape of matrix """

import numpy as np
def np_transpose(matrix):
    """ transposes matrix """
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
transposed_matrix = np.transpose(matrix)
    return transposed_matrix