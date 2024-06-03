#!/usr/bin/env python3
""" Inception Block """


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ A python function that builds an inception block
    as described in Going Deeper with Convolutions (2014) """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = tf.keras.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)
    
    # 1x1 convolution followed by 3x3 convolution
    conv3x3_reduce = tf.keras.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv3x3 = tf.keras.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(conv3x3_reduce)
    
    # 1x1 convolution followed by 5x5 convolution
    conv5x5_reduce = tf.keras.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv5x5 = tf.keras.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(conv5x5_reduce)
    
    # 3x3 max pooling followed by 1x1 convolution
    maxpool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    maxpool_conv = tf.keras.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(maxpool)
    
    # Concatenate all the filters
    output = tf.keras.layers.concatenate([conv1x1, conv3x3, conv5x5, maxpool_conv], axis=-1)
    
    return output
