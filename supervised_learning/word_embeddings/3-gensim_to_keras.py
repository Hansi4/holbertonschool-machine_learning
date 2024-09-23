#!/usr/bin/env python3
""" Extract Word2Vec Module """
import tensorflow as tf


def gensim_to_keras(model):
    # Get the vocabulary size and embedding dimension
    vocab_size = len(model.wv)
    embedding_dim = model.wv.vector_size
    
    # Get the weights from the Gensim model as a TensorFlow tensor
    weights = tf.convert_to_tensor(model.wv.vectors, dtype=tf.float32)

    # Create a Keras Embedding layer with the weights
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True  # Allow further training of the weights
    )
    
    return embedding_layer
