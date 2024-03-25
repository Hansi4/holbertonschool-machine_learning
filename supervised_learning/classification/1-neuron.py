#!/usr/bin/env python3
""" 
Neuron class that defines a single neuron performing binary classification 
"""


import numpy as np


class Neuron:
    """ A single neuron performing binary classification """
    def __init__(self, nx):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self__W = np.random.randn(1, nx)
        self__b = 0
        self__A = 0
