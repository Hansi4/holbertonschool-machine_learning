#!/usr/bin/env python3
"""
Class Deep Neural Network that defines a
deep neural network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """ Defines a deep neural network performing binary classification """
    def __init__(self, nx, layers):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """ Gets the private instance """
        return self.__L

    @property
    def cache(self):
        """ Get method for property b1 """
        return self.__cache

    @property
    def weights(self):
        """ Get method for property b1 """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X 

        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]

            z = np.matmul(W, self.cache["A{}".format(index)]) + b 
            A = 1 / (1 + np.exp(-z))

            self.cache["A{}".format(index + 1)] = A 

        return A, self.cache
