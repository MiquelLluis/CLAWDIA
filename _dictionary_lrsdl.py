from dictol.LRSDL import LRSDL
import numpy as np

from . import util


class DictionaryLRSDL(LRSDL):
    def fit(self, X, *, y_true, l_atoms, step, iterations, offset=0,
                threshold=0, random_seed=None, verbose=False):
        """Train with specific crops and filtering on the training samples.

        Train the dictionary splitting the strains in X into sliced windows
        of length equal to the length of de dictionary.

        PARAMETERS
        ----------
        X: 2d-array, shape=(samples, features)
            Training samples, with equal or more features than the atoms'.

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
        minimum_windows = self.k * n_classes + self.k0
        if X_filtered.shape[0] < minimum_windows:
            raise ValueError(
                "there are not enough training samples for the requested "
                "dimensions of the dictionary. Either try to lower the 'treshold' "
                "parameter or provide more training samples."
            )

        # Train the dictionary
        np.random.seed(random_seed)
        super().fit(X_filtered.T, y_filtered, iterations=iterations, verbose=verbose)

    def predict(self, X, *, threshold=0, offset=0, with_losses=False):
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
        i1 = i0 + self.D.shape[0]
        X_cut = X[:,i0:i1]
        X_cut /= np.linalg.norm(X_cut, axis=1, keepdims=True)

        # E: losses of all strains, shape: (class, strain)
        y_pred, E = super().predict(X_cut.T, loss_mat=True)

        losses = np.min(E, axis=0)
        discarded = losses >= threshold
        y_pred[discarded] = -1

        return (y_pred, losses) if with_losses else y_pred
