import numpy as np

def entropy_node(y):
    y = np.array(y)
    if len(y) == 0:
        return 0.0

    values, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()

    # Stable computation: ignore zero probabilities
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))