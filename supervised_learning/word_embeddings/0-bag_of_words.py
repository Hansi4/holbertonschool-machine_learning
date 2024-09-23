#!/usr/bin/env python3
""" Bag Of Words Module """
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """ A python function that creates a bag of words embedding matrix """
    
    # Regex patterns
    clean_pattern = re.compile(r"[^a-zA-Z0-9\s]")
    single_char_pattern = re.compile(r"\b\w{1}\b")
    
    if vocab is None:
        vocab = []
        for sentence in sentences:
            # Clean sentence and split into words
            cleaned_sentence = clean_pattern.sub(" ", sentence.lower())
            cleaned_sentence = single_char_pattern.sub("", cleaned_sentence)
            vocab.extend(cleaned_sentence.split())
        vocab = sorted(set(vocab))

    # Create a mapping from word to index for efficient lookups
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    
    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            cleaned_word = clean_pattern.sub("", word.lower())
            cleaned_word = single_char_pattern.sub("", cleaned_word).strip()
            if cleaned_word in word_to_index:
                embeddings[i][word_to_index[cleaned_word]] += 1

    return embeddings.astype(int), vocab
