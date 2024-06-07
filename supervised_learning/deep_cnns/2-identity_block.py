#!/usr/bin/env python3
""" Identity Block """


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """ A python function that builds an identity block
    as described in Deep Residual Learning for Image Recognition (2015) """

    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activation = K.activations.relu

    C11 = K.layers.Conv2D(filters=F11,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=activation,
                          kernel_initializer=init)(A_prev)

    Batch_Normalization_11 = K.layers.BatchNormalization(axis=3)(C11)
    Re_LU_11 = K.layers.Activation(activation)(Batch_Normalization_11)

    C33 = K.layers.Conv2D(filters=F3,
                          kernel_size=(3, 3),
                          padding='same',
                          activation=activation,
                          kernel_initializer=init)(Re_LU_11)

    Batch_Normalization_33 = K.layers.BatchNormalization(axis=3)(C33)
    Re_LU_33 = K.layers.Activation(activation)(Batch_Normalization_33)

    C12 = K.layers.Conv2D(filters=F12,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=activation,
                          kernel_initializer=init)(Re_LU_33)

    Batch_Normalization_12 = K.layers.BatchNormalization(axis=3)(C12)

    Addition = K.layers.Add()([Batch_Normalization_12, A_prev])

    output = K.layers.Activation(activation)(Addition)

    return output
