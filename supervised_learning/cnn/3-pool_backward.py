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
                    
                    
                    
                    if mode == 'avg':
                
                    elif mode == 'max':
                

    return dA_prev
