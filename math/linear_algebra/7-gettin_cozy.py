#!/usr/bin/env python3
""" that concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        cat_matrix = []
        for row in mat1:
            cat_matrix.append(list(row))
        for row in mat2:
            cat_matrix.append(list(row))
        return cat_matrix
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_matrix = []
        for i in range(len(mat1)):
            cat_matrix.append(list(mat1[i]) + list(mat2[i]))
        return cat_matrix 
    else:
        return None 
