#!/usr/bin/env python3
""" A function to also save the best iteration of the model """


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """ A python function to also save
    the best iteration of the model """
    callback = []

    if early_stopping is True and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        callback.append(early_stop)

    if learning_rate_decay and validation_data:

        def scheduler(epochs):
            lr = alpha / (1 + decay_rate * epochs)
            return lr

        inv_time_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )

        callback.append(inv_time_decay)

    if save_best:
        save_best_model = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )

        callback.append(save_best_model)
    
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)

    return history
