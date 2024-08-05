#!/usr/bin/env python3
""" Bag Of Words Module """
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """ A python function that creates a bag of words embedding matrix """
    if vocab is None:
        vocab = []
        for sentence in sentences:
            vocab.extend(re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split())
        vocab = sorted(list(set(vocab)))

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            word = re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", word.lower())).strip()
            if word in vocab:
                embeddings[i][vocab.index(word)] += 1

    return embeddings.astype(int), vocab
