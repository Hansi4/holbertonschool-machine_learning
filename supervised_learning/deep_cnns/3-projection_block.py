#!/usr/bin/env python3
""" Projection Block """


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """ A python function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015) """

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)
    activation = K.activations.relu

    C11 = K.layers.Conv2D(filters=F11,
                          kernel_size=(1, 1),
                          padding='same',
                          strides=s,
                          kernel_initializer=init)(A_prev)

    Batch_Normalization_11 = K.layers.BatchNormalization(axis=3)(C11)
    ReLU_11 = K.layers.Activation(activation)(Batch_Normalization_11)

    C33 = K.layers.Conv2D(filters=F3,
                          kernel_size=(3, 3),
                          padding='same',
                          kernel_initializer=init)(ReLU_11)

    Batch_Normalization_33 = K.layers.BatchNormalization(axis=3)(C33)
    ReLU_33 = K.layers.Activation(activation)(Batch_Normalization_33)

    C12 = K.layers.Conv2D(filters=F12,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(ReLU_33)

    Batch_Normalization_12 = K.layers.BatchNormalization(axis=3)(C12)

    SC = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=s,
                         kernel_initializer=init)(A_prev)

    Batch_Normalization_SC = K.layers.BatchNormalization(axis=3)(SC)

    Addition = K.layers.Add()([Batch_Normalization_12, Batch_Normalization_SC])

    output = K.layers.Activation(activation)(Addition)

    return output
