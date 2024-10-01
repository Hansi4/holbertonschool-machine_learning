#!/usr/bin/env python3
""" Train Word2Vec Module """
import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ A python function that creates , builds
    and trains a gensim word2vec model """

    # Define the Word2Vec parameters
    sg = 0 if cbow else 1  # CBOW is 0, Skip-gram is 1
    
    # Create the Word2Vec model
    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   sg=sg,  # Choose between CBOW or Skip-gram
                                   negative=negative,
                                   seed=seed,
                                   workers=workers)
    
    # Train the model over the specified number of epochs
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    
    return model
