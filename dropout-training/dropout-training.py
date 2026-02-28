import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x, dtype=float)

    if p == 0.0:
        pattern = np.ones_like(x)
        return x, pattern

    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)

    # Keep with probability (1 - p)
    keep = rand < (1.0 - p)

    # Pattern: 0 for dropped, 1/(1-p) for kept
    pattern = keep.astype(float) / (1.0 - p)

    output = x * pattern

    return output, pattern