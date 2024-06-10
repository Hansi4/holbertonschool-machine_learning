#!/usr/bin/env python3
""" Dense Block """


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ A python function that builds a dense block as described
    in Densely Connected Convolutional Networks """

    init = K.initializers.he_normal(seed=0)
    H = X

    for _ in range(layers):
        # 1x1 conv
        X = K.layers.BatchNormalization(axis=3)(H)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate * 4, (1, 1),
                            padding='same',
                            kernel_initializer=init)(X)

        # 3x3 conv
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, (3, 3),
                            padding='same',
                            kernel_initializer=init)(X)

        # concatenate all outputs of dense block
        H = K.layers.Concatenate(axis=3)([H, X])
        nb_filters += growth_rate

    return H, nb_filters
