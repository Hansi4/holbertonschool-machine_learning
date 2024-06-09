#!/usr/bin/env python3
""" Dense Block """


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ A python function that builds a dense block as described
    in Densely Connected Convolutional Networks """

    init = K.initializers.he_normal(seed=0)
    activation = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))

    
   

    return model
