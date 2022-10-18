import warnings

import numpy as np


def extract_patches_1d(signals, patch_size, wave_pos=None, n_patches=None, random_state=None,
                       step=1, l2_normed=False, patch_min=1, allow_allzeros=True):
    """TODO:

    signals: ndarray
        If 2d-array must be in fortran order with shape (l_signal, n_signals).

    allow_allzeros: when extracting random patches, if False and l2_normed=True,
        generate another random window position until the l2 norm is != 0.
    
    """
    if signals.ndim > 2:
        raise ValueError("'signals' must be 2d-array at most")
    if signals.ndim == 1:
        signals = signals[:,None]

    rng = np.random.default_rng(random_state)
    l_signals, n_signals = signals.shape
    max_pps = (l_signals - patch_size) / step + 1  # Maximum patches per signal
    if not max_pps.is_integer() and wave_pos is None:
        warnings.warn(
            "'signals' cannot be fully divided into patches, the last"
            f" {(max_pps-1)*step % step:.0f} bins of each signal will be left out",
            RuntimeWarning
        )
    max_pps = int(max_pps)

    # Compute the maximum number of patches that can be extracted and the
    # limits from where to extract patches for each signal.
    if wave_pos is None:
        window_limits = [(0, l_signals-patch_size+1)] * n_signals
        max_patches = max_pps * n_signals
    else:
        window_limits = []
        max_patches = 0
        for p0, p1 in wave_pos:
            p0 += patch_min - patch_size
            p1 -= patch_min
            if p0 < 0:
                p0 = 0
            if p1 + patch_size >= l_signals:
                p1 = l_signals - patch_size
            window_limits.append((p0, p1))
            max_patches += int(np.ceil((p1-p0)/step))

    if n_patches is None:
        n_patches = max_patches
    elif n_patches > max_patches:
        raise ValueError(
            f"the keyword argument 'n_patches' ({n_patches}) exceeds"
            f" the maximum number of patches that can be extracted ({max_patches})."
        )
    
    patches = np.empty((patch_size, n_patches), order='F')

    # Extract all possible patches.
    if n_patches == max_patches:
        k = 0
        for i in range(n_signals):
            p0, p1 = window_limits[i]
            for j in range(p0, p1, step):
                patches[:,k] = signals[j:j+patch_size,i]
                k += 1
    # Extract a limited number of patches randomly selected.
    # <---
    elif l2_normed and not allow_allzeros:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            signal_aux = signals[j:j+patch_size,i]
            while not signal_aux.any():
                j = rng.integers(*window_limits[i])
                signal_aux = signals[j:j+patch_size,i]
            patches[:,k] = signal_aux
    else:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            patches[:,k] = signals[j:j+patch_size,i]
    # --->

    # Normalize each patch to its L2 norm
    if l2_normed:
        patches /= np.linalg.norm(patches, axis=0)

    return patches


def reconstruct_from_patches_1d(patches, step, keepdims=False):
    l_patches, n_patches = patches.shape
    total_len = (n_patches - 1) * step + l_patches
    
    reconstructed = np.zeros(total_len)
    normalizer = np.zeros_like(reconstructed)
    for i in range(n_patches):
        reconstructed[i*step:i*step+l_patches] += patches[:,i]
        normalizer[i*step:i*step+l_patches] += 1
    normalizer[i*step+l_patches:] = 1
    reconstructed /= normalizer

    return reconstructed if not keepdims else reconstructed.reshape(-1,1)
