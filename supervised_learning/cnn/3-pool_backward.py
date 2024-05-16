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
                    # define vertical/horizontal start and end
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'avg':
                        # mean of derivatives would be added to all
                        # cells within the kernel grid in every moves
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                                np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        a_prev_slice \
                            = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]
                            
    return dA_prev