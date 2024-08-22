#!/usr/bin/env python3
""" Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """ A python function that calculates
    the unigram BLEU score for a sentence """

    Brevity_Penalty = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))

    Precision = max([sum(match in reference for match in set(sentence))
                     for reference in references]) / len(sentence)

    BLEU_Score = Brevity_Penalty * np.exp(np.log(Precision))

    return BLEU_Score
