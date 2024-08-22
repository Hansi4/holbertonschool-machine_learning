#!/usr/bin/env python3
""" Cumulative N-gram BLEU score """
import numpy as np


def cumulative_bleu(references, sentence, n):
    """ A python function that calculates
    the cumulative n-gram BLEU score for a sentence """

    Brevity_Penalty = min(1, np.exp(1 -
                          len(min(references, key=len)) / len(sentence)))
    Precision = []

    for m in range:
        n_grams = []
        for reference in references:
            n_grams_ref = []
            for i in range(len(sentence) - (m - 1)):
                if any(sentence[i:i + m] == reference[j:j+m]
                       for j in range(len(reference) - (m - 1))) and \
                        sentence[i:i+m] not in n_grams_ref:
                    n_grams_ref.append(sentence[i:i+m])
            n_grams.append(len(n_grams_ref))
        Precision.append(max(n_grams) / (i + 1))

    BLEU_Score = Brevity_Penalty * np.exp(np.mean(np.log(Precision)))

    return BLEU_Score
