"""
Downsampling utilities for temporal alignment of word-level model embeddings
with discrete brain measurements (e.g., fMRI TRs).

Lanczos and sinc interpolation adapted from litcoder_core, originally based on:
    https://github.com/HuthLab/encoding-model-scaling-laws

Multiple downsampling methods are available:
    - lanczos: Lanczos (windowed sinc) interpolation — smooth low-pass filter
    - average: Average embeddings of words within each TR
    - last: Take the last word's embedding per TR
    - sum: Sum embeddings of words within each TR
"""

import numpy as np


def lanczosfun(cutoff, t, window=3):
    """Compute the Lanczos kernel (windowed sinc function).

    Args:
        cutoff: Cutoff frequency.
        t: Time offsets (scalar or array).
        window: Number of lobes in the Lanczos window.

    Returns:
        Lanczos kernel values at each time offset.
    """
    t = t * cutoff
    with np.errstate(divide='ignore', invalid='ignore'):
        val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi ** 2 * t ** 2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def lanczos_downsample(data, data_times, tr_times, window=3, cutoff_mult=1.0):
    """Downsample using Lanczos interpolation.

    Resamples irregularly-sampled word-level data to regularly-spaced TR times
    using a Lanczos (windowed sinc) low-pass filter. Words closer to a TR get
    higher weight; words beyond the window get zero weight.

    Args:
        data: (n_words, n_features) word-level embeddings.
        data_times: (n_words,) onset time of each word in seconds.
        tr_times: (n_trs,) acquisition time of each TR in seconds.
        window: Number of lobes in the Lanczos window (default 3).
        cutoff_mult: Multiplier for the cutoff frequency (default 1.0).

    Returns:
        (n_trs, n_features) downsampled embeddings aligned with tr_times.
    """
    cutoff = 1 / np.mean(np.diff(tr_times)) * cutoff_mult

    sincmat = np.zeros((len(tr_times), len(data_times)))
    for i in range(len(tr_times)):
        sincmat[i, :] = lanczosfun(cutoff, tr_times[i] - data_times, window)

    # Normalize rows so weights sum to 1 (weighted average)
    # row_sums = sincmat.sum(axis=1, keepdims=True)
    # row_sums[row_sums == 0] = 1  # avoid division by zero for TRs with no nearby words
    # sincmat = sincmat / row_sums

    return np.dot(sincmat, data)


def average_downsample(data, data_times, tr_times, **kwargs):
    """Downsample by averaging embeddings within each TR window.

    Args:
        data: (n_words, n_features) word-level embeddings.
        data_times: (n_words,) onset time of each word in seconds.
        tr_times: (n_trs,) acquisition time of each TR in seconds.

    Returns:
        (n_trs, n_features) averaged embeddings per TR.
    """
    tr = np.mean(np.diff(tr_times))
    output = np.zeros((len(tr_times), data.shape[1]))

    for i, t in enumerate(tr_times):
        mask = (data_times >= t) & (data_times < t + tr)
        if np.any(mask):
            output[i] = np.mean(data[mask], axis=0)

    return output


def last_downsample(data, data_times, tr_times, **kwargs):
    """Downsample by taking the last word's embedding per TR.

    Args:
        data: (n_words, n_features) word-level embeddings.
        data_times: (n_words,) onset time of each word in seconds.
        tr_times: (n_trs,) acquisition time of each TR in seconds.

    Returns:
        (n_trs, n_features) last embedding per TR.
    """
    tr = np.mean(np.diff(tr_times))
    output = np.zeros((len(tr_times), data.shape[1]))

    for i, t in enumerate(tr_times):
        mask = (data_times >= t) & (data_times < t + tr)
        if np.any(mask):
            last_idx = np.where(mask)[0][-1]
            output[i] = data[last_idx]

    return output


def sum_downsample(data, data_times, tr_times, **kwargs):
    """Downsample by summing embeddings within each TR window.

    Args:
        data: (n_words, n_features) word-level embeddings.
        data_times: (n_words,) onset time of each word in seconds.
        tr_times: (n_trs,) acquisition time of each TR in seconds.

    Returns:
        (n_trs, n_features) summed embeddings per TR.
    """
    tr = np.mean(np.diff(tr_times))
    output = np.zeros((len(tr_times), data.shape[1]))

    for i, t in enumerate(tr_times):
        mask = (data_times >= t) & (data_times < t + tr)
        if np.any(mask):
            output[i] = np.sum(data[mask], axis=0)

    return output


_METHODS = {
    'lanczos': lanczos_downsample,
    'average': average_downsample,
    'last': last_downsample,
    'sum': sum_downsample,
}


def downsample(data, data_times, tr_times, method='lanczos', **kwargs):
    """Downsample word-level data to TR-level using the specified method.

    Args:
        data: (n_words, n_features) word-level embeddings.
        data_times: (n_words,) onset time of each word in seconds.
        tr_times: (n_trs,) acquisition time of each TR in seconds.
        method: Downsampling method — 'lanczos', 'average', 'last', or 'sum'.
        **kwargs: Method-specific parameters:
            - lanczos: window (int, default 3), cutoff_mult (float, default 1.0)

    Returns:
        (n_trs, n_features) downsampled data aligned with tr_times.

    Raises:
        ValueError: If method is not supported.
    """
    if method not in _METHODS:
        raise ValueError(f"Unsupported downsampling method: '{method}'. "
                         f"Available: {list(_METHODS.keys())}")

    data = np.asarray(data, dtype=np.float64)
    data_times = np.asarray(data_times, dtype=np.float64)
    tr_times = np.asarray(tr_times, dtype=np.float64)

    return _METHODS[method](data, data_times, tr_times, **kwargs)
