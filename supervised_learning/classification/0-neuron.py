#!/usr/bin/env python3
""" that defines a single neuron performing binary classification """

import numpy as np

class Neuron:
    def __init__(self, nx):
        """ that defines a single neuron performing binary classification """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
    
        self.W = np.random.randn(nx)
        self.b = 0
        self.A = 0

    
    def W(self):
        return self.W

    def b(self):
        return self.b

    def A(self):
        return self.A