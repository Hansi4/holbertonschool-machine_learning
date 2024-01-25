#!/usr/bin/env python3
""" defines function that calculates the shape of matrix """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """

    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
    