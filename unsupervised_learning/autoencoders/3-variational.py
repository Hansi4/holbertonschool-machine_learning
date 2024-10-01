#!/usr/bin/env python3
""" Variational Autoencoder """
import tensorflow as tf
from tensorflow.keras import layers, models, losses, backend as K


def sampling(args):
    """Reparameterization trick by sampling from an isotropic Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    inputs = layers.Input(shape=(input_dims,))
    x = inputs
    
    # Creating hidden layers in the encoder
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    
    # Latent space representation - mean and log variance
    z_mean = layers.Dense(latent_dims, activation=None)(x)
    z_log_var = layers.Dense(latent_dims, activation=None)(x)
    
    # Sampling layer (using reparameterization trick)
    z = layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    
    # Encoder model
    encoder = models.Model(inputs, [z, z_mean, z_log_var], name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dims,))
    x = latent_inputs
    
    # Creating hidden layers in the decoder (reverse of encoder)
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    
    # Output layer in the decoder
    outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    
    # Decoder model
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    
    # VAE Model
    vae_outputs = decoder(encoder(inputs)[0])
    auto = models.Model(inputs, vae_outputs, name='autoencoder')
    
    # Losses - Reconstruction loss and KL Divergence
    reconstruction_loss = losses.binary_crossentropy(inputs, vae_outputs)
    reconstruction_loss *= input_dims
    
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    
    # Compile the model with Adam optimizer and binary cross-entropy loss
    auto.compile(optimizer='adam')
    
    return encoder, decoder, auto
