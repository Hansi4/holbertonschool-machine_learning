#!/usr/bin/env python3
""" Extract Word2Vec Module """
import tensorflow as tf


def gensim_to_keras(model):
    # Get the vocabulary and the weights from the Gensim model
    vocab_size = len(model.wv)
    embedding_dim = model.wv.vector_size
    
    # Create a TensorFlow variable to hold the weights
    weights = tf.Variable(tf.zeros((vocab_size, embedding_dim)), trainable=False)
    
    # Populate the weights matrix using TensorFlow
    for i, word in enumerate(model.wv.index_to_key):
        weights[i].assign(model.wv[word])
    
    # Create a Keras Embedding layer with trainable weights
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=embedding_dim,
                                                embeddings_initializer=tf.keras.initializers.Constant(weights.numpy()),
                                                trainable=True)
    
    return embedding_layer
