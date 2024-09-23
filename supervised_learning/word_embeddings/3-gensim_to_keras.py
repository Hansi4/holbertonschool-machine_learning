#!/usr/bin/env python3
""" Extract Word2Vec Module """
import numpy as np
from keras.layers import Embedding


def gensim_to_keras(model):
    # Get the vocabulary and the weights from the Gensim model
    vocab_size = len(model.wv)
    embedding_dim = model.wv.vector_size
    weights = np.zeros((vocab_size, embedding_dim))
    
    # Populate the weights matrix
    for i, word in enumerate(model.wv.index_to_key):
        weights[i] = model.wv[word]
    
    # Create a Keras Embedding layer with trainable weights
    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[weights],
                                trainable=True)
    
    return embedding_layer
