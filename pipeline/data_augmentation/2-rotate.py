#!/usr/bin/env python3
""" Defines function that rotates an image 90 degrees counter-clockwise """


import tensorflow as tf


def rotate_image(image):
    """ A python function that rotates an image by
    90 degrees counter-clockwise """
    return (tf.image.rot90(image))
