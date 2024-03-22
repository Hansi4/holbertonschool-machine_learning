#!/usr/bin/env python3
""" that defines a single neuron performing binary classification """

import numpy as np

class Neuron:
    def __init__(self, nx):
        """ that defines a single neuron performing binary classification """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
    
        self.W = np.random.randn(nx)
        self.b = 0
        self.A = 0

    
    def get_W(self):
        return self.W

    def get_b(self):
        return self.b

    def get_A(self):
        return self.A