import numpy as np


def dominated_by(v1, v2, epsilon=None):  # Weakly
    v11 = np.asarray(v1)
    v22 = np.asarray(v2)

    if epsilon is not None:
        return np.all(v11 <= v22 + epsilon)
    return np.all(v1 <= v2)
