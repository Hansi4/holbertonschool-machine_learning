#!/usr/bin/env python3
"""
Class Neural Network that defines a neural network with one hidden layer performing binary classification
"""


import numpy as np


class NeuralNetwork:
    """ Defines a neural network with one hidden layer performing binary classification """
    def __init__(self, nx, nodes):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = 0
        self.A1 = 0
        self.W2 = np.random.randn(nodes, nx)
        self.b2 = 0
        self.A2 = 0
