#!/usr/bin/env python3
""" Save and load configuration """


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Saves a models weights """
    network.save_weights(filename=filename, save_format=save_format)


def load_weights(network, filename):
    """ Loads a model’s weights """
    network.load_weights(filepath=filename)
