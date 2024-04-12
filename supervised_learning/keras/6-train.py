#!/usr/bin/env python3
""" A function to also train the model using early stopping """


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ A python function to also train the model using early stopping """
    callback = []

    if early_stopping is True and validation_data is not None:
        callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history
