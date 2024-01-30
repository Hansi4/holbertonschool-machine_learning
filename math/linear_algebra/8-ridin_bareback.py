#!/usr/bin/env python3
""" that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ performs matrix multiplication """
    mat1_rows, mat1_columns = len(mat1), len(mat1[0])
    mat2_rows, mat2_columns = len(mat2), len(mat2[0])

    if mat1_columns != mat2_rows:
        return None
    
    result = []
    for i in range(mat1_rows):
        result.append([])
        for j in range(mat2_columns):
            dot = sum(mat1[i][k] * mat2[k][j] for k in range(mat1_columns))
            result[i].append(dot)
    
    return result
