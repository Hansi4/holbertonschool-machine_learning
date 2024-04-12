#!/usr/bin/env python3
""" Save and load configuration """


import tensorflow.keras as K


def save_config(network, filename):
    """ Saves a model’s configuration in JSON format """
    network_config = network.to_json()
    with open(filename, "w") as f:
        f.write(network_config)

def load_config(filename):
    """ Loads a model with a specific configuration """
    with open(filename, "r") as f:
        load_network_config = f.read()
    return K.models.model_from_json(load_network_config)
