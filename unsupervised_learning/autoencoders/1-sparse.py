#!/usr/bin/env python3
""" Sparse Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ A python function that creates a sparse autoencoder """

    # Encoder
    input_layer = keras.layers.Input(shape=(input_dims,))
    encoded = input_layer

    # Adding hidden layers to the encoder with ReLU activation
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Latent space representation
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=
                                keras.regularizers.l1(lambtha))(encoded)

    # Decoder
    decoded = latent

    # Adding hidden layers to the decoder (reverse of encoder)
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer with Sigmoid activation for binary cross-entropy
    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoded)

    # Models
    encoder = keras.models.Model(input_layer, latent, name="encoder")
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_output = decoder_input

    # Build decoder layers
    for nodes in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(nodes,
                                            activation='relu')(decoder_output)
    decoder_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_output)

    decoder = keras.models.Model(decoder_input,
                                 decoder_output, name="decoder")

    # Full autoencoder model
    auto = keras.models.Model(input_layer,
                              decoder(encoder(input_layer)),
                              name="autoencoder")

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
