#!/usr/bin/env python3
""" that defines a single neuron performing binary classification """

import numpy as np


def __init__(self, nx):
    """ that defines a single neuron performing binary classification """
    if not isinstance(nx, int) or nx <= 0:
        raise TypeError("nx must be an integer")
    if not isinstance(x, int) or int(x) < 1:
        raise ValueError("nx must be a positive integer")
    