"""
Split-half reliability ceiling for the LeBel benchmark.

Uses repeated trials of the same story (same subject, multiple sessions)
to estimate the noise ceiling via split-half correlation.
"""

import numpy as np
import pickle
from glob import glob
from tqdm import tqdm

from brainscore_core.metrics import Score


def rowwise_pearson(A, B):
    """
    Compute Pearson correlation row-wise between two matrices.
    A, B: shape (n_rows, n_cols)
    Returns r: shape (n_rows,)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A0 = A - np.nanmean(A, axis=1, keepdims=True)
    B0 = B - np.nanmean(B, axis=1, keepdims=True)
    num = np.nansum(A0 * B0, axis=1)
    den = np.sqrt(np.nansum(A0 ** 2, axis=1) * np.nansum(B0 ** 2, axis=1))
    r = np.where(den == 0.0, np.nan, num / den)
    return r


def split_half_ceiling(trial_paths, story_name, lead_trim=10, n_splits=500, seed=42):
    """
    Compute split-half reliability ceiling from repeated trial recordings.

    Args:
        trial_paths: list of paths to pickle files, each containing
                     a dict with {story_name: array(n_trs, n_voxels)}
        story_name: the story to compute ceiling for
        lead_trim: number of leading TRs to trim from each trial
        n_splits: number of random split-half iterations
        seed: random seed for reproducibility

    Returns:
        Score with median ceiling across voxels, with per-voxel ceilings in attrs['raw']
    """
    # Load all trials
    brain_data_trials = []
    for path in tqdm(trial_paths, desc='loading trials'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        brain_data_trials.append(data[story_name][lead_trim:])

    brain_data_trials = np.stack(brain_data_trials)  # (n_trials, n_trs, n_voxels)
    # 10 trials, 251 ytrs, 20484 voxels
    n_trials = brain_data_trials.shape[0]

    # Split-half reliability
    rng = np.random.RandomState(seed)
    voxel_ceilings = []
    for _ in tqdm(range(n_splits), desc='split-half ceiling'):
        perm = rng.permutation(n_trials)
        # we split into two halves
        half = n_trials // 2
        # averaging reduces noise - cleaner estimate of the 'true' brain response
        # (5, 251, 20484).mean(0) ->  (251, 20484)
        group1 = brain_data_trials[perm[:half]].mean(axis=0)  # (n_trs, n_voxels)
        group2 = brain_data_trials[perm[half:]].mean(axis=0)
        # transpose so we have voxels across rows
        # correlate each voxel between the two group averages - > 20484 correlation values
        corrs = rowwise_pearson(group1.T, group2.T)  # (n_voxels,)
        voxel_ceilings.append(corrs)

    # avg the 20484-length correlation vector across 500 splits
    # so we will have one final correlation per voxel
    # and we take the median across voxels -> a single ceiling scalar
    voxel_ceilings = np.nanmean(voxel_ceilings, axis=0)  # (n_voxels,)
    voxel_ceilings = np.sqrt(2 * voxel_ceilings / (1 + voxel_ceilings))
    median_ceiling = np.nanmedian(voxel_ceilings)

    ceiling = Score(median_ceiling)
    ceiling.attrs['raw'] = voxel_ceilings
    return ceiling


def load_ceiling(trial_dir, story_name='wheretheressmoke', subject='UTS03',
                 lead_trim=40, n_splits=500, seed=42):
    """
    Convenience function to compute ceiling from a directory of trial pickle files.

    Args:
        trial_dir: directory containing noslice_sub-{subject}_ses-*_{story}_surface.pkl files
        story_name: the repeated story name
        subject: subject identifier
        lead_trim: number of leading TRs to trim
        n_splits: number of split-half iterations
        seed: random seed

    Returns:
        Score with ceiling value
    """
    pattern = f"{trial_dir}/noslice_sub-{subject}_ses-*_{story_name}_surface.pkl"
    trial_paths = sorted(glob(pattern))
    if not trial_paths:
        raise FileNotFoundError(f"No trial files found matching: {pattern}")

    print(f"Found {len(trial_paths)} repeated trials for {story_name}")
    return split_half_ceiling(trial_paths, story_name, lead_trim=lead_trim,
                              n_splits=n_splits, seed=seed)
