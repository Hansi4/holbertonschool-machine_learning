#!/usr/bin/env python3
""" Pooling Forward Prop """


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ A python function that performs forward
    propagation over a pooling layer of a neural network """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # output size
    output_height = int((h_prev - kh) / sh + 1)
    output_width = int((w_prev - kw) / sw + 1)

    # initialize output
    pooled_img = np.zeros((m, output_height, output_width, c_prev))

    for i in range(output_height):
        for j in range(output_width):
            image_zone = A_prev[:, i * sh : i * sh + kh,
                                j * sw : j * sw + kw, :]
        if mode == 'max':
            pooled_img[:, i, j, :] = np.max(image_zone, axis=(1, 2))
        elif mode == 'avg':
            pooled_img[:, i, j, :] = np.average(image_zone, axis=(1, 2))

    return pooled_img
