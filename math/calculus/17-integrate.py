#!/usr/bin/env python3
""" A script that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ A function that calculates the integral of a polynomial """
    if not isinstance(C, int) or not isinstance(poly, list) or len(poly) == 0:
        return None
    integral = [C]
    for power, coefficient in enumerate(poly):
        if (coefficient % (power + 1)) == 0:
            new_coefficient = coefficient // (power + 1)
        else:
            new_coefficient = coefficient / (power + 1)
        integral.append(new_coefficient)
    while integral[-1] == 0 and len(integral) > 1:
        integral = integral[:-1]
    return integral
