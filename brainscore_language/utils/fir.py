"""
FIR (Finite Impulse Response) delay stacking for hemodynamic response modeling.

The BOLD fMRI signal peaks ~4-6s after a neural event. With TR=2s, this means
the brain response to a word heard at TR t is best captured at TRs t+1 through t+4.

FIR delay stacking creates shifted copies of the feature matrix at multiple delays
and concatenates them, allowing regression to learn the hemodynamic response shape.

Reference: Huth et al. (encoding-model-scaling-laws)
"""

import numpy as np


def apply_fir_delays(data, n_delays=4):
    """Apply FIR delay stacking to downsampled embeddings.

    Creates delayed copies of the input feature matrix (shifts 1..n_delays TRs)
    and concatenates them along the feature axis. Row t of delay-d output contains
    features from row t-d of the input, with zero-padding for early TRs.

    Args:
        data: (n_trs, n_features) numpy array, e.g. output of Lanczos downsampling.
        n_delays: Number of delays to stack (default 4, giving delays 1,2,3,4).

    Returns:
        (n_trs, n_features * n_delays) numpy array with delay-stacked features.
    """
    data = np.asarray(data)
    n_trs, n_features = data.shape
    delayed = np.zeros((n_trs, n_features * n_delays), dtype=data.dtype)

    for d in range(1, n_delays + 1):
        # Delay d: row t gets features from row t-d
        delayed[d:, (d - 1) * n_features : d * n_features] = data[:-d] if d < n_trs else data[:0]

    return delayed
