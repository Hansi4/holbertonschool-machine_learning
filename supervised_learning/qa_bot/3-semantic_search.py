#!/usr/bin/env python3
""" Semantic Search """


import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """ A python function that performs
    semantic search on a corpus of documents """

    # 1. Start with a list containing just the sentence that will be compared against the corpus
    documents = [sentence]

    # 2. Load and Read Documents
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())

    # 3. Load Pre-trained Model
    """
        The Universal Sentence Encoder (USE) model from TensorFlow Hub encodes sentences into fixed-size embeddings. 
        This model is designed to produce semantically meaningful vectors that capture the meaning of sentences.
    """
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Pass the list of documents to the model. This produces embeddings for each document, including the sentence
    embeddings = model(documents)

    # Compute the similarity between each pair of embeddings using the inner product
    correlation = np.inner(embeddings, embeddings)

    # Find the Most Similar Document, Identify Most Similar Document
    closest = np.argmax(correlation[0, 1:])

    # Return the Most Similar Document
    similar = documents[closest + 1]

    return similar
