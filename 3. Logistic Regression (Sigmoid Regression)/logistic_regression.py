import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z
    (z = w*x + b)

    Args:
        z (np.ndarray): A scalar, numpy array of any size.

    Returns:
        g (np.ndarray): sigmoid(z), with the same shape as z
         
    """
    g: np.ndarray = 1 / (1 + np.exp(-z))
    return g
