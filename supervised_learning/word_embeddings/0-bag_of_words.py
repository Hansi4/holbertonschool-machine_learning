#!/usr/bin/env python3
""" Bag Of Words Module """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ A python function that creates a bag of words embedding matrix """

    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
