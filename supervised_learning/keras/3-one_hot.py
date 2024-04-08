#!/usr/bin/env python3
""" A function that
converts a label vector into a one-hot matrix """


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ A python function that converts
    a label vector into a one-hot matrix """

    one_matrix = K.utils.to_categorical(labels, classes)
    return one_matrix
