#!/usr/bin/env python3
""" FastText Module """
import gensim

def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ A python function that creates and trains a gensim FastText model """

    model = gensim.models.FastText(sentences=sentences, vector_size=size, 
                                   min_count=min_count, window=window, 
                                   negative=negative, epochs=iterations,
                                   seed=seed, workers=workers, sg=not cbow)

    return model
