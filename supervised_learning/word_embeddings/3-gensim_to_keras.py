#!/usr/bin/env python3
""" Extract Word2Vec Module """
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Embedding


def gensim_to_keras_embedding(model):
    """ A python function that converts a gensim
    word2vec model to a keras Embedding layer """
    word_vectors = model.wv
    vocab_size, vector_size = word_vectors.vectors.shape
    embedding_matrix = np.zeros((vocab_size, vector_size))
    
    for word, index in word_vectors.key_to_index.items():
        embedding_matrix[index] = word_vectors[word]
    
    keras_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[embedding_matrix],
        trainable=True
    )
    
    return keras_embedding
