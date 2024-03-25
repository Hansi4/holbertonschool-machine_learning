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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Get method for property Weights """
        return self.__W

    @property
    def b(self):
        """ Get method for property Bias """
        return self.__b

    @property
    def A(self):
        """ Get method for activation function """
        return self.__A
