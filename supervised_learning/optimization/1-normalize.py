#!/usr/bin/env python3
""" Normalize """


import numpy as np


def normalize(X, m, s):
    """ A python function that normalizes (standardizes) a matrix """
    Z = (X - m) / s

    return Z
