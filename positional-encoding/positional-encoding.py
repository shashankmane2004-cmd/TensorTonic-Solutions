import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len)[:, np.newaxis]          # (seq_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]               # (1, d_model)

    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    angles = positions * angle_rates

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])  # even indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])  # odd indices

    return pe