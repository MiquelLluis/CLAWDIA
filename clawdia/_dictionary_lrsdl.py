import dictol.LRSDL
import numpy as np
from time import time

from . import lib


class DictionaryLRSDL(dictol.LRSDL.LRSDL):
    """Interface for the Low-Rank Shared Dictionary Learning class.

    Attributes
    ----------
    t_train : float
        Training time in seconds.

    lambd : float
        See self.__init__() for details.
    
    lambd2 : float
        See self.__init__() for details.
    
    eta : float
        See self.__init__() for details.
    
    D : ndarray
        Class-specific dictionary.
    
    X : ndarray
        Class-specific coefficient vector of the training set given when
        calling self.fit().
    
    Y : ndarray
        Class-specific target vector (the training set) given when calling
        self.fit().
    
    k : int
        See self.__init__() for details.
    
    k0 : int
        See self.__init__() for details.
    
    updateX_iters : int
        See self.__init__() for details.
    
    updateD_iters : int
        See self.__init__() for details.
    
    D_range : list[int]
        Auxiliar list containing the range of indices of each class in D.
    
    D0 : ndarray
        Shared dictionary.
    
    Y_range : list[→nt]
        Auxiliar list containing the range of indices of each class in Y.
        Derived directly from 'train_label', equivalent to the 'y_true' labels.
        Example: given train_label = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], then
        Y_range = [0, 4, 6]. The first value is always 0, marking the
        start of the first class, and the last value is always the number of
        classes + 1.
    
    X0 : ndarray
        Shared coefficient vector of the training set given when calling
        self.fit().
    
    
    Notes
    -----
    The authors of Dictol didn't provide a seed parameter for the random
    initialisation of the dictionary. If reproducibility is important, one must
    set the global numpy's seed before callint ``LRSDL.__init__()``.


    References
    ----------
    .. [1] Vu, T. H.; Monga, V. (2017). *Fast low-rank shared dictionary
           learning for image classification*, IEEE Transactions on Image
           Processing, 26(11), 5160–5175.
           (https://doi.org/10.1109/TIP.2017.2729885)

    """
    def __init__(self, lambd=0.01, lambd2=0.01, eta=0.0001, k=10, k0=5,
                 updateX_iters=100, updateD_iters=100):
        r"""Initialize the LRSDL dictionary.

        This method sets up the parameters required for training class-specific
        and shared dictionaries. These dictionaries are used to represent data
        with sparsity and low-rank properties, which can be regularized by the
        parameters defined below.

        Parameters
        ----------
        lambd : float
            Regularisation parameter for the sparsity term:

            .. math::

               \lambda \|X\|_1

            This encourages sparsity in the class-specific dictionary, similar
            to the LASSO regularisation term.

        lambd2 : float
            Regularisation parameter for the reconstruction term:

            .. math::

               \frac{\lambda_2}{2} \|X^0 - M^0\|^2

            This ensures that the shared vector (used to select shared atoms)
            is sparse and close to the mean shared vector, ensuring consistency
            across all :math:`X^0`.

        eta : float
            Regularisation parameter for the low-rank term:

            .. math::

               \eta \| D^0 \|_*

            Here, :math:`\|\cdot\|_*` is the `nuclear norm
            <https://encyclopediaofmath.org/wiki/Nuclear_norm>`_, which
            enforces the shared dictionary to have low rank.

        k : int
            Number of class-specific atoms for each class. The total number of
            atoms in the class-specific dictionary is given by :math:`k \times
            C`, where :math:`C` is the number of classes.

        k0 : int
            Total number of shared atoms. A value of :math:`k_0 = 0` indicates
            that no shared dictionary is used.

        updateX_iters, updateD_iters : int
            These parameters are passed to the parent class
            :meth:`LRSDL.__init__`. However, they are suspected to be **dummy
            parameters** because no usage of them could be found in the
            original implementation. They are retained here for compatibility
            but appear to have no functional effect in this class.

        Warnings
        --------
        The `updateX_iters` and `updateD_iters` parameters are inherited from the 
        parent class :class:`LRSDL`, but they appear to be unused in this 
        implementation. Consider verifying their relevance before relying on them.

        Notes
        -----
        - The parameters `lambd`, `lambd2`, and `eta` control the sparsity and
          low-rank properties of the dictionaries.
        - Setting ``k0 = 0`` disables the shared dictionary.

        """
        super().__init__(
            lambd=lambd, lambd2=lambd2, eta=eta, k=k, k0=k0,
            updateX_iters=updateX_iters, updateD_iters=updateD_iters
        )

        self.t_train = None
    
    def __str__(self):
        params = [
            f"lambd={self.lambd}",
            f"lambd2={self.lambd2}",
            f"eta={self.eta}",
            f"k={self.k}",
            f"k0={self.k0}",
            f"updateX_iters={self.updateX_iters}",
            f"updateD_iters={self.updateD_iters}"
        ]
        params_str = ", ".join(params)
        state = []
        
        # Training time
        if self.t_train is not None:
            state.append(f"Training time: {self.t_train:.2f} sec")
        else:
            state.append("Not trained")
        
        # Dictionary and coefficients
        state_attrs = [
            ('D', 'D'),
            ('D0', 'D0'),
            ('X', 'X'),
            ('X0', 'X0'),
            ('Y', 'Y'),
            ('Y_range', 'Y_range'),
            ('D_range', 'D_range')
        ]
        for attr_name, display_name in state_attrs:
            attr = getattr(self, attr_name, None)
            if attr is not None:
                if isinstance(attr, np.ndarray):
                    if attr_name in ('Y_range', 'D_range'):
                        state.append(f"{display_name}: {attr}")
                    else:
                        state.append(f"{display_name}: shape={attr.shape}")
                elif isinstance(attr, list):
                    state.append(f"{display_name}: length={len(attr)}")
                else:
                    state.append(f"{display_name}: {attr}")
            else:
                state.append(f"{display_name}: Not initialized")
        
        state_str = "\n  ".join(state)
        
        return f"DictionaryLRSDL({params_str})\n  {state_str}"

    def fit(self, X, *, y_true, l_atoms, iterations, step=None,
            threshold=0, random_seed=None, verbose=False, show_after=5):
        """Train the LRSDL dictionary.

        This method trains the dictionary using the provided data and allows
        for several configuration options:
        
        - Split the input data `X` into sliding windows of length `l_atoms`.
        - Use the entire input as a single window.
        - Discard training windows whose L2-norm is below a specified
          threshold.

        The splitting behavior depends on the `step` parameter. If `step` is
        `None`, the entire input is treated as a single window. Otherwise,
        overlapping patches of size `l_atoms` are created with the specified
        step size.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples. The number of features must be equal to or
            greater than the dictionary's atom size.

        y_true : ndarray of shape (n_samples,)
            Labels corresponding to the samples in `X`. The length of `y_true`
            must equal the number of samples in `X`.

        l_atoms : int
            Length of the dictionary's atoms.

        iterations : int
            Number of training iterations.

        step : int, optional
            The step size for splitting input samples into patches of length
            `l_atoms`. If not specified, it is set the same (`step = l_atoms`)
            so that all information abailable is extracted from `X` without any
            repetition (overlap).

        threshold : float, optional
            L2-norm threshold (relative to the maximum L2-norm in each strain).
            Training windows with L2-norm below this value will be discarded.
            Default is 0 (only null arrays are discarded).

        random_seed : int, optional
            Random seed for reproducibility. Default is `None`.

        verbose : bool, optional
            If True, print verbose output during training. Default is False.

        show_after : int, optional
            If `verbose` is True, progress will be displayed every `show_after`
            iterations. Default is 5.

        Returns
        -------
        `None`
            The method trains the dictionary in place.

        Notes
        -----
        Per-class sufficiency is now checked on the *effective* number of
        training windows (after patching and thresholding), not on the raw
        number of input strains per class.

        """
        if not isinstance(y_true, np.ndarray):
            raise TypeError("'y_true' must be a numpy array.")
        
        if X.shape[1] < l_atoms:
            raise ValueError("X must have at least 'l_atoms' features.")

        # Sort `X` and `y_true` so that labels are consecutive and begin with 1.
        i_sorted = np.argsort(y_true)
        y_true = y_true[i_sorted]
        X = X[i_sorted]

        if y_true[0] != 1:
            raise ValueError("labels in 'y_true' must be integers starting from 1")

        if step is None:
            step = l_atoms

        n_x, l_x = X.shape
        n_wps = int((l_x - l_atoms) / step + 1)  # Number of windows per strain
        if n_wps <= 0:
            raise ValueError(
                "No windows can be extracted with the given 'l_atoms' and 'step'."
            )
        y_windowed = np.repeat(y_true, n_wps).reshape(n_x, n_wps)
        
        # Split X -> X_windowed:
        X_windowed = np.empty((n_x, n_wps, l_atoms), dtype=float)
        for ix in range(n_x):
            X_windowed[ix] = lib.extract_patches(X[ix], patch_size=l_atoms, step=step)


        # Filter windows: Discard those which their L2-norm is lower than the
        # specified by the relative threshold:
        norms = np.linalg.norm(X_windowed, axis=2)         # (n_x, n_wps)
        l2_maxs = np.max(norms, axis=1, keepdims=True)     # (n_x, 1)
        m_keep = (norms > l2_maxs*threshold)              # (n_x, n_wps)  Mask of windows to keep per signal

        X_filtered = X_windowed[m_keep]  # (n_filtered, l_atoms)
        y_filtered = y_windowed[m_keep]  # (n_filtered)

        if verbose:
            n_keep = np.sum(m_keep)
            n_out = m_keep.size - n_keep
            frac_keep = n_keep / m_keep.size
            print(f"filtered: {n_out}\t kept: {n_keep} ({frac_keep:.1%})")

        # ---- Check: enough windows to build the dictionary ----
        n_classes = len(set(y_true))
        # Ensure all classes are represented with minlength.
        per_class_counts = np.bincount(y_filtered, minlength=n_classes + 1)[1:]  # drop 0-bin
        least_eff = int(np.min(per_class_counts))
        if least_eff < self.k:
            i_class = int(np.argmin(per_class_counts) + 1)
            raise ValueError(
                "insufficient effective training windows after patching/thresholding: "
                f"class {i_class} has {per_class_counts[i_class-1]} < k={self.k}. "
                "Consider lowering 'threshold', using a smaller 'step' (more overlap), "
                "or providing more training data."
            )
        # Also ensure global sufficiency for class-specific + shared atoms:
        minimum_windows = self.k * n_classes + self.k0
        if X_filtered.shape[0] < minimum_windows:
            raise ValueError(
                "there are not enough training samples for the requested "
                "dimensions of the dictionary. Either try to lower the 'threshold' "
                "parameter, reduce 'step' to extract more patches, or provide "
                "more training samples."
            )

        # Train the dictionary
        np.random.seed(random_seed)
        tic = time()
        super().fit(
            X_filtered.T, y_filtered, iterations=iterations, verbose=verbose, show_after=show_after
        )
        tac = time()
        
        self.t_train = tac - tic

    def _predict(self, Y, loss_mat=False):
        """Adapted from DICTOL's LRSDL.predict method."""
        N = Y.shape[1]
        E = np.zeros((self.num_classes, N))
        for c in range(self.num_classes):
            # Dc in D only
            Dc_ = dictol.utils.get_block_col(self.D, c, self.D_range)
            # Dc in D and D0
            Dc = np.hstack((Dc_, self.D0)) if self.k0 > 0 else Dc_
            lasso = dictol.optimize.Lasso(Dc, lambd=self.lambd)
            lasso.fit(Y)
            Xc = lasso.solve()
            residual_matrix = Y - np.dot(Dc, Xc)
            E[c, :] = (0.5 * np.sum(residual_matrix*residual_matrix, axis=0)
                       + self.lambd * np.sum(np.abs(Xc), axis=0))
        pred = np.argmin(E, axis=0) + 1
        
        return (pred, E) if loss_mat else pred

    def predict(self, X, *, threshold=0, offset=0, with_losses=False):
        """Predict the class of each window in X.

        The class of a window is the class of the closest codeword to that
        window in the dictionary.

        Parameters
        ----------
        X : 2d-array, shape=(n_signals, n_samples)
            Input signals, with equal or more samples than the atoms'.

        threshold : float, optional
            Loss threshold ABOVE which signals will be marked as "unknown" class,
            which corresponds to the label value -1.
            Zero by default, all signals will be classified.

        offset : int, optional
            Index i0 at which to crop the input signals X.
            The i1 will be `offset + l_atoms`. By default 0.

        with_losses : bool, optional
            If True, return a tuple with the class predictions and the
            corresponding losses.

        Returns
        -------
        y_pred : 1d-array, shape=(n_signals)
            Class predictions for each input signal.

        losses : 1d-array, shape=(n_signals), optional
            Losses of the closest codewords to each input signal.
            Only returned if with_losses=True.

        """
        # Cut signals to dico's length and discard the rest:
        i0 = offset
        i1 = i0 + self.D.shape[0]
        X_cut = X[:,i0:i1]
        with np.errstate(divide='ignore', invalid='ignore'):
            X_cut /= np.linalg.norm(X_cut, axis=1, keepdims=True)

        # E: losses of all strains, shape: (class, strain)
        y_pred, E = self._predict(X_cut.T, loss_mat=True)

        losses = np.min(E, axis=0)
        
        if threshold != 0:
            discarded = losses >= threshold
            y_pred[discarded] = -1

        return (y_pred, losses) if with_losses else y_pred

    def save(self, file: str) -> None:
        """Save the dictionary to a file.

        Save the dictionary attributes using NumPy's 'np.savez()'.
        
        Parameters
        ----------
        file : str
            Path to the file where to save the dictionary.
        
        """
        vars_ = vars(self)
        np.savez(file, **vars_)

