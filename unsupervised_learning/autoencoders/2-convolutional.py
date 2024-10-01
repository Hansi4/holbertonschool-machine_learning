#!/usr/bin/env python3
""" Convolutional Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ A python function that creates a convolutional autoencoder """

    # Encoder
    input_layer = keras.layers.Input(shape=input_dims)
    encoded = input_layer
    for f in filters:
        encoded = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoded)

    encoder = keras.models.Model(input_layer, encoded)

    decoder_input = keras.layers.Input(shape=latent_dims)
    decoder_output = decoder_input
    for f in reversed(filters[1:]):
        decoder_output = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(decoder_output)
        decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)

    decoder_output = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu')(decoder_output)
    decoder_output = keras.layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same',
        activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_outputs = encoder(encoder_input)
    auto_outputs = decoder(auto_outputs)
    autoencoder = keras.models.Model(encoder_input, auto_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
