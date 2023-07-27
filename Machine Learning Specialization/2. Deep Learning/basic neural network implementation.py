import numpy as np


def dense(a_in: np.array, W: np.array, b: np.array) -> np.array:
    """
    Dense layer implementation
    :param a_in: input array
    :param W: weights
    :param b: bias
    :return: output array
    """
    units = W.shape[1]
    
    assert a_in.shape[1] == W.shape[0], f"Input shape {a_in.shape} does not match weights shape {W.shape}"
    assert b.shape[0] == units, f"Bias shape {b.shape} does not match weights shape {W.shape}"

    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(a_in, w) + b[j]
        a_out[j] = 1 / (1 + np.exp(-z))
    
    return a_out


def dense_vectorized(a_in: np.array, W: np.array, b: np.array) -> np.array:
    """
    Dense layer implementation
    :param a_in: input array
    :param W: weights
    :param b: bias
    :return: output array
    """
    units = W.shape[1]
    
    assert a_in.shape[1] == W.shape[0], f"Input shape {a_in.shape} does not match weights shape {W.shape}"
    assert b.shape[0] == units, f"Bias shape {b.shape} does not match weights shape {W.shape}"

    z = np.matmul(a_in, W) + b
    a_out = 1 / (1 + np.exp(-z))
    
    return a_out
