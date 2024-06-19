#!/usr/bin/env python3
""" Transition Layer """


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """ A python function that builds a transition layer as
    described in Densely Connected Convolutional Networks """

    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    # 1x1 conv
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters,
                        (1, 1),
                        padding='same',
                        kernel_initializer=init)(X)

    # average pool
    X = K.layers.AveragePooling2D((2, 2),
                                  strides=(2, 2),
                                  padding='same')(X)

    return X, nb_filters