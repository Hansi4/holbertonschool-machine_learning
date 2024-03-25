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
        """ Get method for property activation function """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        return self.A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1/m) * (-(m_loss))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return [prediction, cost]
