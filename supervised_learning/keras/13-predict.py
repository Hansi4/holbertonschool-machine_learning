#!/usr/bin/env python3
""" Make prediction using neural network """


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ A function that makes a prediction using a neural network """
    return network.predict(
        x=data,
        verbose=verbose
    )
