#!/usr/bin/env python3
""" Defines function that performs a random crop of an image """


import tensorflow as tf


def crop_image(image, size):
    """ A python function that performs a random crop of an image """
    return (tf.image.random_crop(image, size=size))
