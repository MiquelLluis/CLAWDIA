from dictol.LRSDL import LRSDL
import numpy as np
from time import time

from . import lib


class DictionaryLRSDL(LRSDL):
    """Interface for the Low-Rank Shared Dictionary Learning class.

    NOTE: The authors of Dictol didn't provide a seed parameter for the random
    initialization of the dictionary. If reproducibility is important, one must
    set the global numpy's seed before callint LRSDL.__init__().

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


    References
    ----------
    Vu, T. H.; Monga, V. (2017). Fast low-rank shared dictionary learning for image classification.
    IEEE Transactions on Image Processing, 26(11), 5160–5175. https://doi.org/10.1109/TIP.2017.2729885

    """
    def __init__(self, lambd=0.01, lambd2=0.01, eta=0.0001, k=10, k0=5,
                 updateX_iters=100, updateD_iters=100):
        """Initialize the dictionary.
        
        Parameters
        ----------
        lambd : float
            Regularization term:
                lambd * ||X||_1
            Makes the class-specific vector sparse, symilar to the LASSO
            regularization term.

        lambd2 : float
            Regularization term:
                lambd2 / 2 * ||X⁰-M⁰||²
            Makes the shared vector (selection of shared atoms) sparse and close to
            the mean shared vector, i.e. all {X⁰} close between them.

        eta : float
            Regularization term:
                eta * ||D⁰||_*
            Enforces the shared dictionary to be low-rank.

        k : int
            Number of class-specific atoms for each class. The total number of
            atoms in the class-specific dictionary is then `k * nc` where 'nc' is
            the number of classes.
        
        k0 : int
            Total number of shared atoms. k0=0 is equivalent to the case when there
            is no shared dictionary.
        
        updateX_iters, updateD_iters : int
            *I think they are not used in this class at all.*
        
        """
        super().__init__(
            lambd=lambd, lambd2=lambd2, eta=eta, k=k, k0=k0,
            updateX_iters=updateX_iters, updateD_iters=updateD_iters
        )

        self.t_train = None

    def fit(self, X, *, y_true, l_atoms, iterations, step=None,
            threshold=0, random_seed=None, verbose=False, show_after=5):
        """Train de LRSDL dictionary.

        Train the dictionary allowing several options:
        - Split the strains in X into sliced windows of length equal to the
          length of de dictionary, or
        - use the whole strain as a window.
        - Discard training windows whose L2-norm is below a threshold.

        Parameters
        ----------
        X : 2d-array, shape=(samples, features)
            Training samples, with equal or more features than the atoms'.

        y_true : np.ndarray
            Labels of samples in X, with `len(y_true) == X.shape[0]`.

        l_atoms : int
            Lenght of the atoms of the dictionary.

        iterations : int
            Number of training iterations.

        step : int, optional
            For splitting strains in X into the specified 'l_atoms' in order to
            generate the training patches.
            No splitting by default.

        threshold : float, optional
            L2-norm threshold relative to the window of max(L2-norm) of each
            strain, below which to discard the rest of the reconstruction windows.
            No threshold by default.

        verbose : bool
            If True, increase verbosity of LRSDL.fit().
        
        show_after : int, optional
            If verbose is True, show the progress every 'show_after' iterations.

        """
        if not isinstance(y_true, np.ndarray):
            raise TypeError("'y_true' must be a numpy array.")
        # Check that there are at least 'self.k' samples of each class in the
        # training set.
        _least_samples = np.min(np.bincount(y_true)[1:])
        if _least_samples < self.k:
            i_class = np.argmin(np.bincount(y_true)[1:]) + 1
            raise ValueError(
                f"there are less than {self.k} samples of class {i_class} in"
                f" the training set ()"
            )

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
        y_windowed = np.repeat(y_true, n_wps).reshape(n_x, n_wps)
        
        # Split X -> X_windowed:
        X_windowed = np.empty((n_x, n_wps, l_atoms), dtype=float)
        for ix in range(n_x):
            X_windowed[ix] = lib.extract_patches(X[ix].T, patch_size=l_atoms, step=step).T


        # Filter windows: Discard those which their L2-norm is lower than the
        # specified by the relative threshold:
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
        tic = time()
        super().fit(
            X_filtered.T, y_filtered, iterations=iterations, verbose=verbose, show_after=show_after
        )
        tac = time()
        
        self.t_train = tac - tic

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
        y_pred, E = super().predict(X_cut.T, loss_mat=True)

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

