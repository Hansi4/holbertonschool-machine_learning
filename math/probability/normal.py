#!/usr/bin/env python3
""" Class Normal that represents a normal distribution """


class Exponential:
    """ Class Normal that represents a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ represents a normal distribution """
        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)

                summation = 0
                for x in data:
                    summation += (x - mean) ** 2
                stddev = (summation / len(data)) ** (1/2)
                self.stddev = stddev

    def z_score(self, x):
        """ """
        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """ """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        calculates the value of the pdf of a given x-value
        :param x: the x-value
        :return: pdf value for x
        """
        π = 3.1415926536
        e = 2.7182818285
        return ((1 / (self.stddev * ((2 * π) ** 0.5))) *
                (e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)))
