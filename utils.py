import numpy as np


def dominated_by(v1, v2, epsilon=None):  # Weakly
    if epsilon is not None:
        return np.all(v1 <= v2 + epsilon)
    return np.all(v1 <= v2)
