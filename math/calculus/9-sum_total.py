#!/usr/bin/env python3
""" that calculates sum for i=1:n of i**2 """


def summation_i_squared(n):
    """ calculates sum for i=1:n of i**2 """
    sum = (n*(n+1)*(2*n+1)) / 6
    return sum 