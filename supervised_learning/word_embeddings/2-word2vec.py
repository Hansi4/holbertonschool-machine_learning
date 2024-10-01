#!/usr/bin/env python3
""" Train Word2Vec Module """
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ A python function that creates , builds
    and trains a gensim word2vec model """

    # Choose CBOW (if cbow=True) or Skip-gram (if cbow=False)
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        min_count=min_count, 
        window=window, 
        negative=negative, 
        sg=sg, 
        epochs=iterations, 
        seed=seed, 
        workers=workers)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
