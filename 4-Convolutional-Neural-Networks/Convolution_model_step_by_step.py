import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(2)

def zero_pad(X, pad):
    # X dimension is (m, n_h, n_w, n_c)
    # X_pad dimension is (m, n_h + 2*pad, n_w + 2*pad, n_c)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values = (0, 0))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s) 
    Z = Z + b
    return Z

def conv_forward(A_prev, W, b, h_parameters):
    stride = h_parameters['stride']
    pad = h_parameters['pad']

    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (f, f, n_c_prev, n_c) = W.shape

    n_h = int((n_h_prev + 2 * pad - f) / stride + 1)
    n_w = int((n_w_prev + 2 * pad - f) / stride + 1)

    Z = np.zeros((m, n_h, n_w, n_c))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_h):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_w):
                hori_start = w * stride
                hori_end = hori_start + f

                for c in range(n_c):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, hori_start:hori_end, :]

                    weights = W[:, :, :, c]
                    bias = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, bias)

    assert(Z.shape == (m, n_h, n_w, n_c))

    cache = (A_prev, W, b, h_parameters)

    return Z, cache

def pool_forward(A_prev, h_parameters, mode = 'max'):
    f = h_parameters['f']
    stride = h_parameters['stride']

    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape

    n_h = int((n_h_prev - f) / stride + 1)
    n_w = int((n_w_prev - f) / stride + 1)
    n_c = n_c_prev

    A = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_w):
                hori_start = w * stride
                hori_end = hori_start + f
                for c in range(n_c):
                    a_slice_prev = a_prev[vert_start:vert_end, hori_start:hori_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)
                    if mode == 'average':
                        A[i, h, w, c] = np.average(a_slice_prev)

    cache = (A_prev, h_parameters)

    assert(A.shape == (m, n_h, n_w, n_c))

    return A, cache
