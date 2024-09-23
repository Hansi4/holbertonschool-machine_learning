#!/usr/bin/env python3
""" FastText Module """
import gensim

def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """ A python function that creates and trains a gensim FastText model """
    return gensim.models.FastText(sentences=sentences, size=vector_size, 
                                   min_count=min_count, window=window, 
                                   negative=negative, epochs=epochs,
                                   seed=seed, workers=workers, sg=not cbow)
