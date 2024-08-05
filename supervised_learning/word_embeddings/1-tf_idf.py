#!/usr/bin/env python3
""" TF-IDF Module """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ A python function that creates a TF-IDF embedding """

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
