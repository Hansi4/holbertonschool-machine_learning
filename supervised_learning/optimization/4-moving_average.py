#!/usr/bin/env python3
""" Moving Average """


import numpy as np


def moving_average(data, beta):
    """ A python function that calculates
    the weighted moving average of a data set """
    m_av = []
    w = 0

    for i, d in enumerate(data):
        # update weight average
        w = beta * w + (1 - beta) * d
        # bias correction
        w_new = w / (1 - beta **(i + 1))
        # add moving average to list
        m_av.append(w_new)

    return m_av
