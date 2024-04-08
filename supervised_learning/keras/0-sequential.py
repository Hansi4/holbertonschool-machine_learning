#!/usr/bin/env python3
""" A function that builds
a neural network with the Keras library """


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ A python function that builds
    a neural network with the Keras library """
    model = K.Sequential()

    for i in range(len(layers)):
        model.add(
            K.layers.Dense(
                layers[i], 
                activation=activations[i], 
                kernel_regularizer=K.regularizers.L2(lambtha), 
                input_dim=nx))

        if i != len(layers) - 1 and keep_prob is not None:
            model.add(K.layers.Dropout(1-keep_prob))
    return model
