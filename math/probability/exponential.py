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
            self.lambtha = lambtha
