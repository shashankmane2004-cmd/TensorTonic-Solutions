import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
        N = len(seqs)
        L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    
    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len

    padded = np.full((N, L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        padded[i, :length] = seq[:length]

    return padded