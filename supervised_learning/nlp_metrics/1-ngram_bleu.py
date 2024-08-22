#!/usr/bin/env python3
""" N-gram BLEU score """
import numpy as np


def ngram_bleu(references, sentence, n):
    """ A python function that calculates
    the n-gram BLEU score for a sentence """

    Brevity_Penalty = min(1, np.exp(1 -
                          len(min(references, key=len)) / len(sentence)))
    n_grams = []
    n_grams_ref = 0

    for reference in references:
        n_grams_ref = []
        for i in range(len(sentence) - (n - 1)):
            if any(sentence[i:i + n] == reference[j:j+n]
                   for j in range(len(reference) - (n - 1))) and \
                    sentence[i:i+n] not in n_grams_ref:
                n_grams_ref.append(sentence[i:i+n])
        n_grams.append(len(n_grams_ref))

    Precision = max(n_grams) / (i + 1)

    BLEU_Score = Brevity_Penalty * np.exp(np.log(Precision))

    return BLEU_Score
