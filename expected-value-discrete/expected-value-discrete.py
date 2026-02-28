import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError("x and p must have same shape")

    if np.any(p < 0) or not np.isclose(np.sum(p), 1.0):
        raise ValueError("invalid probabilities")

    return float(np.sum(x * p))