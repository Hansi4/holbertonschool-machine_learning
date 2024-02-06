#!/usr/bin/env python3
""" A script that calculates sum for i=1:n of i**2 """


def summation_i_squared(n):
    """ A function that calculates sum for i=1:n of i**2 """
    if type(n) is not int:
        return None
    sum = (n*(n+1)*(2*n+1)) / 6
    return sum
