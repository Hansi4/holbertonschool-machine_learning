#!/usr/bin/env python3
""" DenseNet-121 """


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ A python function that builds the DenseNet-121 architecture as
    described in Densely Connected Convolutional Networks """

    X = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    init = K.initializers.he_normal

    
   

    return model
