#!/usr/bin/env python3
""" A function to also train the model with learning rate decay """


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ A python function to also train the model
    with learning rate decay """
    callback
    if
    
    
    
    
    
    
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history
