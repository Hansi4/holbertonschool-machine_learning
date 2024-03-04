#!/usr/bin/env python3
""" A script that loads data from a file as a pd.DataFrame """


import pandas as pd


def from_file(filename, delimiter):
    """ A function that loads data from a file as a pd.DataFrame """
    data = pd.read_csv(filename, delimiter=delimiter)
    df = pd.DataFrame(data)
    return df