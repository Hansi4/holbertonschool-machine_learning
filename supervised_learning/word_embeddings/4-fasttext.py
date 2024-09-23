#!/usr/bin/env python3
""" FastText Module """
import genism


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """ A python function that creates and trains a genism fastText model """

    return genism.models.FastText(sentences=sentences, vector_size=vector_size, min_count=min_count, 
                    window=window, negative=negative, epochs=epochs,
                    seed=seed, workers=workers, sg=not cbow)
