#!/usr/bin/env python3
""" Pooling Back Prop """


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ A python function that performs back propagation
    over a pooling layer of a neural network """

    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # initialize shape for dA_prev and dW
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):

                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_prev_slice \ 
                            = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, f]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, f] +=\
                            mask * dA[i, h, w, f]
                    elif mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, f] += (
                                np.ones((kh, kw)) * avg_dA)

    return dA_prev
