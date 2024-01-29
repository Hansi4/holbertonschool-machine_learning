#!/usr/bin/env python3
""" defines function that calculates the shape of matrix """

import numpy as np
def np_transpose(matrix):
    """ transposes matrix """
mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
transposed_matrix = np.transpose(matrix)
    return transposed_matrix