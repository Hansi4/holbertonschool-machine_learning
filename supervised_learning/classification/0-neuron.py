#!/usr/bin/env python3
""" that defines a single neuron performing binary classification """

import numpy as np


def __init__(self, nx):
    """ that defines a single neuron performing binary classification """
    if not isinstance(nx, int):
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")
    