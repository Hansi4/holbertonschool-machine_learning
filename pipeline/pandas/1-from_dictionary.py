#!/usr/bin/env python3
""" A script that created a pd.DataFrame from a dictionary """


import pandas as pd


def from_dictionary():
    """ A python script that created a pd.DataFrame from a dictionary """
    data = {
        'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']
    }

    index_labels = list("ABCD")

    df = pd.DataFrame(data, index=index_labels)
    return df