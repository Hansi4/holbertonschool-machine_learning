#!/usr/bin/env python3
""" Class Exponential that represents an exponential distribution """


class Exponential:
    """ Class Exponential that represents an exponential distribution """
    def __init__(self, data=None, lambtha=1.):
        """ represents an exponential distribution """
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period """
        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        result = lambtha * (e ** (-lambtha * x))
        return result

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period """
        if x < 0:
            return 0

        e = 2.7182818285
        lambtha = self.lambtha

        result = 1 - (e ** (-lambtha * x))
        return result
