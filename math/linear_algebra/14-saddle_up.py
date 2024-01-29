#!/usr/bin/env python3
""" defines function that calculates the shape of matrix """


import numpy as np

def np_matmul(mat1, mat2):
    """ performs matrix multiplication """
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    result = np_matmul(mat1, mat2)
    return result