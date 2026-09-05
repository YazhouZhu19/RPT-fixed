"""
Dataset Specifics
Extended from ADNet code by Hansen et al.
"""

import random

import numpy as np


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CMR':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'

    elif dataset == 'CHAOST2':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'SPLEEN'
    elif dataset == 'SABS':
        label_names[0] = 'BG'
        label_names[1] = 'SPLEEN'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'GALLBLADDER'
        label_names[5] = 'ESOPHAGUS'
        label_names[6] = 'LIVER'
        label_names[7] = 'STOMACH'
        label_names[8] = 'AORTA'
        label_names[9] = 'IVC'  # Inferior vena cava
        label_names[10] = 'PS_VEIN'  # portal vein and splenic vein
        label_names[11] = 'PANCREAS'
        label_names[12] = 'AG_R'  # right adrenal gland
        label_names[13] = 'AG_L'  # left adrenal gland

    return label_names


def get_folds(dataset):
    """Return ordered evaluation folds.

    The final case in every fold is the dedicated support case.  The preceding
    cases form disjoint query partitions; keeping this as a list is therefore
    important (a set loses the support position).
    """
    FOLD = {}
    if dataset == 'CMR':
        FOLD[0] = list(range(0, 8))
        FOLD[1] = list(range(7, 15))
        FOLD[2] = list(range(14, 22))
        FOLD[3] = list(range(21, 29))
        FOLD[4] = list(range(28, 35)) + [0]
        return FOLD

    elif dataset == 'CHAOST2':
        FOLD[0] = list(range(0, 5))
        FOLD[1] = list(range(4, 9))
        FOLD[2] = list(range(8, 13))
        FOLD[3] = list(range(12, 17))
        FOLD[4] = list(range(16, 20)) + [0]
        return FOLD
    elif dataset == 'SABS':
        FOLD[0] = list(range(0, 7))
        FOLD[1] = list(range(6, 13))
        FOLD[2] = list(range(12, 19))
        FOLD[3] = list(range(18, 25))
        FOLD[4] = list(range(24, 30)) + [0]
        return FOLD
    else:
        raise ValueError(f'Dataset: {dataset} not found')


def get_excluded_slice_indices(labels, excluded_labels):
    """Return slices containing any held-out class (Setting 2)."""
    if excluded_labels is None or len(excluded_labels) == 0:
        return np.empty(0, dtype=np.int64)

    excluded_mask = np.isin(labels, excluded_labels).any(axis=(1, 2))
    return np.flatnonzero(excluded_mask)


def sample_xy(spr, k=0, b=215):
    import torch

    _, h, v = torch.where(spr)

    if len(h) == 0 or len(v) == 0:
        horizontal = 0
        vertical = 0
    else:

        h_min = min(h)
        h_max = max(h)
        if b > (h_max - h_min):
            kk = min(k, int((h_max - h_min) / 2))
            horizontal = random.randint(max(h_max - b - kk, 0), min(h_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            horizontal = random.randint(max(h_min - kk, 0), min(h_max - b + kk, 256 - b - 1))

        v_min = min(v)
        v_max = max(v)
        if b > (v_max - v_min):
            kk = min(k, int((v_max - v_min) / 2))
            vertical = random.randint(max(v_max - b - kk, 0), min(v_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            vertical = random.randint(max(v_min - kk, 0), min(v_max - b + kk, 256 - b - 1))

    return horizontal, vertical
