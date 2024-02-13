#!/usr/bin/env python3
""" Class Exponential that represents an exponential distribution """


class Exponential:
    """ Class Exponential that represents an exponential distribution """
    
    def __init__(self, data=None, lambtha=1.):
        """ represents an exponential distribution """
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            lambtha = float(len(data) / sum(data))


    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period """
        if type(k) is not int:
            k = int(k)
        if (k < 0):
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        factorial = 1
        for i in range(k):
            factorial *= (i + 1)
        pmf = ((e ** -lambtha) * (lambtha ** k)) / factorial
        return pmf