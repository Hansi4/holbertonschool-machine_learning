#!/usr/bin/env python3
""" Identity Block """


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """ A python function that builds an identity block
    as described in Deep Residual Learning for Image Recognition (2015) """

    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activation = K.activations.relu

    F11, F3, F12 = filters
    initializer = he_normal(seed=0)

    # First component of the main path
    X = Conv2D(filters=F11, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=initializer)(A_prev)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second component of the main path
    X = Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third component of the main path
    X = Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = BatchNormalization(axis=3)(X)

    # Final step: Add shortcut value to the main path, and pass it through a RELU activation
    X = Add()([X, A_prev])
    X = Activation('relu')(X)

    return X
