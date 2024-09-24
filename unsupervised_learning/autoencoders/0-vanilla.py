#!/usr/bin/env python3
""" "Vanilla" Autoencoder """
import tensorflow as tf
from tensorflow.keras import layers, models


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ A python function that creates an autoencoder """

    # Encoder
    input_layer = layers.Input(shape=(input_dims,))
    encoded = input_layer
    
    # Adding hidden layers to the encoder with ReLU activation
    for nodes in hidden_layers:
        encoded = layers.Dense(nodes, activation='relu')(encoded)
    
    # Latent space representation
    latent = layers.Dense(latent_dims, activation='relu')(encoded)
    
    # Decoder
    decoded = latent
    
    # Adding hidden layers to the decoder (reverse of encoder)
    for nodes in reversed(hidden_layers):
        decoded = layers.Dense(nodes, activation='relu')(decoded)
    
    # Output layer with Sigmoid activation for binary cross-entropy
    output_layer = layers.Dense(input_dims, activation='sigmoid')(decoded)
    
    # Models
    encoder = models.Model(input_layer, latent, name="encoder")
    decoder_input = layers.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    
    # Build decoder layers
    for nodes in reversed(hidden_layers):
        decoder_output = layers.Dense(nodes, activation='relu')(decoder_output)
    decoder_output = layers.Dense(input_dims, activation='sigmoid')(decoder_output)
    
    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    
    # Full autoencoder model
    auto = models.Model(input_layer, decoder(encoder(input_layer)), name="autoencoder")
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
