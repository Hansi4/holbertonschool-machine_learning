#!/usr/bin/env python3
""" FastText Module """
import gensim
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    # Setting sg to 1 for Skip-gram and 0 for CBOW
    sg = 0 if cbow else 1
    
    # Create the FastText model
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )
    
    # Build vocabulary from sentences
    model.build_vocab(sentences)
    
    # Train the model
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    
    # Return the trained model
    return model
