import numpy as np

from dictol import LRSDL

from . import util


def train_lrsdl(X, *, y_true, l_atoms, step, iterations, init_kwargs, offset=0,
                threshold=0, random_seed=None, verbose=False):
    """Train LRSDL dictionary with specific length of atoms.

    Train the LRSDL dictionary splitting the strains in X into sliced windows
    of length 'l_atoms' equal to the desired length of de dictionary.

    PARAMETERS
    ----------
    X: 2d-array, shape=(samples, features)
        Training samples, with `X.shape[1] >= l_atoms`.

    y_true: array-like
        Labels of samples in X, with `len(y_true) == X.shape[0]`.

    l_atoms: int
        Lenght of the atoms of the dictionary.

    step: int
        For splitting strains in X into the specified 'l_atoms' in order to
        generate the training patches.

    iterations: int
        Number of training iterations.

    offset: int, optional
        Index i0 at which to crop the input strains X.
        The i1 will be `offset + l_atoms`. By default 0.

    threshold: float, optional
        L2-norm threshold relative to the window of max(L2-norm) of each
        strain, below which to discard the rest of the reconstruction windows.
        No threshold by default.

    init_kwargs:
        Key-word arguments passed to LRSDL.__init__().

    verbose: bool
        If True, increase verbosity of LRSDL.fit().

    """
    X_crop = X[:,offset:]

    n_x, l_x = X_crop.shape
    n_wps = int((l_x - l_atoms) / step + 1)  # Number of windows per strain
    y_windowed = np.repeat(y_true, n_wps).reshape(n_x, n_wps)
    
    # Split X -> X_windowed:
    X_windowed = np.empty((n_x, n_wps, l_atoms), dtype=float)
    for ix in range(n_x):
        X_windowed[ix] = util.extract_patches(X_crop[ix].T, patch_size=l_atoms, step=step).T
    
    # Filter windows: Discard those which their L2-norm is lower than the
    # specified by the relative threshold:
    # <---
    norms = np.linalg.norm(X_windowed, axis=2)         # (n_x, n_wps)
    l2_maxs = np.max(norms, axis=1, keepdims=True)     # (n_x, 1)
    m_keep = norms >= l2_maxs*threshold                # (n_x, n_wps)  Mask of windows to keep.

    m_alltrue = np.all(m_keep, axis=1)
    i_ends = np.argmin(m_keep, axis=1, keepdims=True)  # (n_x, 1)
    m_out = i_ends <= np.arange(m_keep.shape[1])      # (n_x, n_wps)
    m_out[m_alltrue] = False
    m_keep = ~m_out

    X_filtered = X_windowed[m_keep]  # (n_filtered, l_atoms)
    y_filtered = y_windowed[m_keep]  # (n_filtered)
    # --->

    if verbose:
        n_out = np.sum(m_out)
        n_keep = np.sum(m_keep)
        frac_keep = n_keep / m_keep.size
        print(f"filtered: {n_out}\t kept: {n_keep} ({frac_keep:.1%})")

    # Check that there are enough windows to build the dictionary:
    n_classes = len(set(y_true))
    minimum_windows = init_kwargs['k']*n_classes + init_kwargs['k0']
    if X_filtered.shape[0] < minimum_windows:
        raise ValueError(
            "there are not enough training samples for the requested "
            "dimensions of the dictionary. Either try to lower the 'treshold' "
            "parameter or provide more training samples."
        )

    # Init and train the dictionary
    dico = LRSDL(**init_kwargs)
    np.random.seed(random_seed)
    dico.fit(X_filtered.T, y_filtered, iterations=iterations, verbose=verbose)

    return dico


def predict_lrsdl(X, dico, *, threshold=0, offset=0, with_losses=False):
    """

    Parameters
    ----------
    threshold: float, optional
        Loss threshold ABOVE which signals will be marked as "unknown" class,
        which corresponds to the label value -1.
        Zero by default, all signals will be classified.

    offset: int, optional
        Index i0 at which to crop the input signals X.
        The i1 will be `offset + l_atoms`. By default 0.

    """
    # Cut signals to dico's length and discard the rest:
    i0 = offset
    i1 = i0 + dico.D.shape[0]
    X_cut = X[:,i0:i1]
    X_cut /= np.linalg.norm(X_cut, axis=1, keepdims=True)
    y_pred, E = dico.predict(X_cut.T, loss_mat=True)  # E: losses of all strains, shape (class, strain)
    losses = np.min(E, axis=0)
    discarded = losses >= threshold
    y_pred[discarded] = -1

    return (y_pred, losses) if with_losses else y_pred
