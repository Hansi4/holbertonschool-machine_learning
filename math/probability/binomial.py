#!/usr/bin/env python3
""" Class Binomial that represents a binomial distribution """


class Binomial:
    """ Class Binomial that represents a binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """ represents a binomial distribution """
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            self.n = n
            if 0 <= p <= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p
