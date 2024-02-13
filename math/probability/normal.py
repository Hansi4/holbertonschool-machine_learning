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
