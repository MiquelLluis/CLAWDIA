import time
import warnings

import numpy as np
import scipy.optimize
import spams
from tqdm import tqdm

from . import estimators
from . import lib


# Remove warning from OpenMP, present in older versions of python-spams.
if not '__version__' in dir(spams) or spams.__version__ <= '2.6.5.4':
    import os
    os.environ['KMP_WARNINGS'] = 'FALSE'


class DictionarySpams:
    """Sparse Dictionary Learning (SDL) model for waveform denoising via SPAMS.

    This class provides an object-oriented implementation of a Sparse
    Dictionary Learning model, designed for the denoising and reconstruction
    of waveforms. At its core, it utilizes the `trainDL` function for
    dictionary learning and the `lasso` function for sparse coding from the
    SPAMS-python library [1]_.

    It extends these core functionalities to arbitrarily long signals and
    minibatch processing for large datasets. Additionally, the class includes
    various utilities for signal preprocessing, composite models of denoising
    (such as iterative reconstruction), and the ability to easily save and
    load the dictionary's state.

    Attributes
    ----------
    dict_init : ndarray
        Atoms of the initial dictionary. Remains unaltered after training.
    components : ndarray
        Atoms of the current (trained) dictionary.
    model : tuple
        SPAMS' trainDL model components in the form (A, B, iter).
    d_size : int
        Number of atoms in the dictionary (dictionary size).
    a_length : int
        Length of each atom in the dictionary (patch size).
    lambda1 : float
        Regularization parameter for training the dictionary.
    batch_size : int
        Batch size used in mini-batch training.
    n_iter : int
        Number of iterations performed during training.
    t_train : float
        Total training time in seconds.
    trained : bool
        Indicates whether the dictionary has been trained.
    n_train : int
        Number of patches used during training.
    mode_traindl : int
        Training mode for SPAMS' `trainDL` function.
    modeD_traindl : int
        Dictionary mode for SPAMS' `trainDL` function.
    mode_lasso : int
        Mode for SPAMS' `lasso` function.
    identifier : str
        Optional identifier or note for distinguishing the dictionary.

    References
    ----------
    .. [1] SPAMS (for python), (http://spams-devel.gforge.inria.fr/).
           Last accessed in October 2018.
    
    """
    def __init__(self,
             dict_init=None,
             model=None,
             signal_pool=None, a_length=None, d_size=None, wave_pos=None,
             patch_min=1, l2_normed=True, allow_allzeros=False,
             random_state=None, ignore_completeness=False,
             lambda1=None, batch_size=64, n_iter=None, n_train=None,
             trained=False, mode_traindl=0, modeD_traindl=0, mode_lasso=2,
             identifier=''):
        """Initialize the dictionary.

        There are two ways to initialize the dictionary:
        
        1. By directly providing the initial dictionary with `dict_init`.
        2. By providing a collection of signals (`signal_pool`) from which
           atoms are randomly extracted to form the initial dictionary.

        If the second option is used, `a_length` and `d_size` must be
        explicitly specified to define the size of the dictionary. Additional
        optional parameters provide more control over this process.

        Parameters
        ----------
        dict_init : ndarray of shape (d_size, a_length), optional
            Atoms of the initial dictionary. If `None`, `signal_pool` must be
            provided.
        model : dict, optional
            SPAMS' `trainDL` model components as a dictionary with elements {A,
            B, iter}. Must be provided if continuing training from a previous
            state.
        signal_pool : ndarray of shape (n_signals, n_samples), optional
            A collection of signals from which atoms are extracted to form the
            initial dictionary. Ignored if `dict_init` is provided.
        a_length : int, optional
            Length of each atom in the dictionary (patch size). Required if
            `signal_pool` is provided.
        d_size : int, optional
            Number of atoms in the dictionary. Required if `signal_pool` is
            provided.
        wave_pos : array-like of shape (n_signals, 2), optional
            Positions of waveforms within `signal_pool` to extract atoms from.
            If `None`, the entire array is used.
        patch_min : int, default=1
            Minimum number of samples for each extracted patch. Ignored if
            `wave_pos` is `None`.
        l2_normed : bool, default=True
            If `True`, normalize extracted atoms to their L2 norm.
        allow_allzeros : bool, default=False
            By default, random atoms with all zeros are excluded from the
            initial dictionary. If `allow_allzeros=True`, they are allowed.
        random_state : int, optional
            Seed for random sampling from `signal_pool`.
        ignore_completeness : bool, optional, default=False
            If `False`, the dictionary must be overcomplete (`d_size >
            a_length`).
        lambda1 : float, optional
            Regularization parameter for training.
        batch_size : int, default=64
            Batch size used during training.
        n_iter : int, optional
            Total number of iterations for training. If `None`, this must be
            set when calling the `train` method.
        n_train : int, optional
            Number of patches used for training. Informational only.
        trained : bool, default=False
            Indicates whether the dictionary is already trained.
        mode_traindl : int, default=0
            Training mode for SPAMS' `trainDL` function. See SPAMS
            documentation.
        modeD_traindl : int, default=0
            Dictionary mode for SPAMS' `trainDL` function. See SPAMS
            documentation.
        mode_lasso : int, default=2
            Mode for SPAMS' `lasso` function. See SPAMS documentation.
        identifier : str, optional
            A note or label for identifying the dictionary.

        Notes
        -----
        This method initializes the dictionary but does not train it. Use the
        `train` method for training.

        """
        self.model = model
        self.dict_init = dict_init
        self.components = dict_init
        self.a_length = a_length
        self.d_size = d_size
        self.lambda1 = lambda1
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.t_train = -n_iter if n_iter is not None and n_iter < 0 else None
        self.trained = trained
        self.n_train = n_train
        self.mode_traindl = mode_traindl
        self.modeD_traindl = modeD_traindl
        self.mode_lasso = mode_lasso
        self.identifier = identifier

        self._check_initial_parameters(signal_pool, ignore_completeness)

        # Explicit initial dictionary (trained or not).
        if self.dict_init is not None:
            self.d_size, self.a_length = self.dict_init.shape

        # Get the initial atoms from a set of signals.
        else:
            self.dict_init = lib.extract_patches(
                signal_pool,
                patch_size=self.a_length,
                limits=wave_pos,
                n_patches=self.d_size,
                l2_normed=l2_normed,
                allow_allzeros=allow_allzeros,
                patch_min=patch_min,
                random_state=random_state
            )
            self.components = self.dict_init

    def train(self, patches, lambda1=None, n_iter=None, warm_start=False,
              verbose=False, threads=-1, **kwargs):
        """Train the dictionary.

        Train the dictionary with the given patches.

        This also allows a warm start using the previous components as initial
        dictionary, but only if the lambda1 parameter is the same. It can be
        thought of as adding more iterations to the training. Hence, providing
        different patches is discouraged and untested.

        Parameters
        ----------
        patches : 2d-array(signals, samples)
            Training patches.

        lambda1 : float, optional
            Regularization parameter of the learning algorithm.
            It is not needed if already specified at initialization.

        n_iter : int, optional
            Total number of iterations to perform.
            If a negative number is provided it will perform the computation
            during the corresponding number of seconds.
            For instance `n_iter = -5` trains the dictionary during 5 seconds.

        warm_start : bool
            If True, use the previous components as initial dictionary.
            It can be thought of as adding more iterations to the training.
            Providing different patches is discouraged and untested.

        verbose : bool, optional
            If True print the iterations (might not be shown in real time).

        threads : int, optional
            Number of threads to use during training, see [1].

        **kwargs
            Passed directly to 'spams.trainDL', see [1].

        See Also
        --------
        clawdia.lib.extract_patches : Useful for generating the training `patches`.

        """
        if self.trained:
            if not warm_start:
                raise ValueError("the dictionary has already been trained")
            if lambda1 is not None and lambda1 != self.lambda1:
                raise ValueError("the 'lambda1' parameter must be the same "
                                 "as the one used at the previous training")

        if patches.shape[1] != self.a_length:
            raise ValueError("the length of 'patches' must be the same as the"
                             " atoms of the dictionary")
        
        if n_iter is None:
            if self.n_iter is None:
                raise TypeError("'n_iter' not specified")
            else:
                n_iter = self.n_iter
            
        if lambda1 is None:
            if self.lambda1 is None:
                raise TypeError("'lambda1' not specified")
            
            lambda1 = self.lambda1

        tic = time.time()
        components, model = spams.trainDL(
            patches.T,            # SPAMS works with Fortran order.
            D=self.components.T,  #
            model=self.model,
            batchsize=self.batch_size,
            K=self.d_size,  # In SPAMS argo, the dictionary size is the number of atoms.
            lambda1=lambda1,
            iter=n_iter,
            mode=self.mode_traindl,
            modeD=self.modeD_traindl,
            verbose=verbose,
            numThreads=threads,
            return_model=True,
            **kwargs
        )
        self.components = components.T
        self.model = model
        tac = time.time()

        if warm_start:
            if n_iter < 0:
                self.n_iter += model['iter']
                self.t_train += -n_iter
            else:
                self.n_iter += n_iter
                self.t_train += tac - tic

        else:
            self.trained = True
            self.lambda1 = lambda1
            self.n_train = patches.shape[0]

            if n_iter < 0:
                self.n_iter = model['iter']
                self.t_train = -n_iter
            else:
                self.n_iter = n_iter
                self.t_train = tac - tic

    def _reconstruct_single(self, signal, sc_lambda, step=1, **kwargs_lasso):
        # TODO: Add kwarg option to disable the patch normalization.
        # This might be usefull for tasks such detection or when a heavy
        # discrimination is needed.
        patches, norms = lib.extract_patches(
            signal,
            patch_size=self.a_length,
            step=step,
            l2_normed=True,
            return_norm_coefs=True
        )
        code = spams.lasso(
            patches.T,            # SPAMS works with Fortran order.
            D=self.components.T,  #
            lambda1=sc_lambda,
            mode=self.mode_lasso,
            **kwargs_lasso
        )
        patches = ((self.components.T @ code) * norms).T

        signal_rec = lib.reconstruct_from_patches_1d(patches, step)

        return signal_rec, code

    def reconstruct(self, signal, sc_lambda, step=1, normed=True, with_code=False, **kwargs):
        """Reconstruct a signal as a sparse combination of dictionary atoms.

        Parameters
        ----------
        signal : ndarray
            Sample to be reconstructed.

        sc_lambda : float
            Regularization parameter of the sparse coding transformation.

        step : int, 1 by default
            Sample interval between each patch extracted from signal.
            Determines the number of patches to be extracted. 1 by default.

        normed : boolean, True by default
            Normalize the result to the maximum absolute value.

        with_code : boolean, False by default.
            If True, also returns the coefficients array.

        **kwargs
            Passed directly to the external learning function.

        Returns
        -------
        signal_rec : array
            Reconstructed signal.

        code : array(a_length, d_size), optional
            Transformed data, encoded as a sparse combination of atoms.
            Returned when 'with_code' is True.

        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("'signal' must be a numpy array")

        signal_rec, code = self._reconstruct_single(signal, sc_lambda, step, **kwargs)

        if normed and signal_rec.any():
            norm = np.max(np.abs(signal_rec))
            signal_rec /= norm
            code /= norm

        return (signal_rec, code) if with_code else signal_rec

    def _reconstruct_batch(self, strains, *, sc_lambda, step=1, normed_windows=True, **kwargs):
        ns = strains.shape[0]

        patches, norms = lib.extract_patches(
            strains, patch_size=self.a_length, step=step, l2_normed=normed_windows,
            return_norm_coefs=True
        )
        codes = spams.lasso(
            patches.T,            # SPAMS works with Fortran order.
            D=self.components.T,  #
            lambda1=sc_lambda,
            mode=self.mode_lasso,
            **kwargs
        )
        
        patches = ((self.components.T @ codes) * norms).T
        lp = patches.shape[1]
        np_ = patches.shape[0] // ns  # Number of patches per strain
        patches = patches.reshape(ns, np_, lp, order='C')

        reconstructions = np.empty_like(strains)
        for i in range(ns):
            reconstructions[i] = lib.reconstruct_from_patches_1d(patches[i], step)

        return reconstructions

    def reconstruct_batch(self, signals, sc_lambda, out=None, step=1, normed=True,
                          verbose=True, **kwargs):
        """TODO

        Reconstruct multiple signals, each one as a sparse combination of
        dictionary atoms.

        WARNING: Only viable for small 'signals' set, it is really memory
        expensive (all patches are stored in a single array in memory).

        WARNING: 'out' deprecated, left for backwards compatibility but will
        be ignored if given.

        """
        out = self._reconstruct_batch(signals, sc_lambda=sc_lambda, step=step, **kwargs)

        if normed and out.any():
            with np.errstate(divide='ignore', invalid='ignore'):
                out /= np.max(np.abs(out), axis=1, keepdims=True)
            np.nan_to_num(out, copy=False)

        return out

    def reconstruct_minibatch(self, signals, *, sc_lambda, step=1, batchsize=4, normed=True,
                              normed_windows=True, verbose=True, **kwargs):
        """TODO

        Reconstruct multiple signals, each one as a sparse combination of
        dictionary atoms. Minibatch version.

        """
        n_signals = signals.shape[0]
        n_minibatch = n_signals // batchsize
        out = np.empty_like(signals)
        loop = range(n_minibatch)
        if verbose:
            loop = tqdm(loop)
        
        for ibatch in loop:
            i0 = ibatch * batchsize
            i1 = i0 + batchsize
            minibatch = signals[i0:i1]
            out[i0:i1] = self._reconstruct_batch(
                minibatch, sc_lambda=sc_lambda, step=step, normed_windows=normed_windows, **kwargs
            )
        if n_minibatch == 0:
            # In case there was no point in using a minibatch:
            i1 = 0

        # If 'n_signals' was not divisible by 'batchsize' reconstruct the
        # remaining signals:
        if i1 < n_signals:
            i0 = i1
            minibatch = signals[i0:]
            out[i0:] = self._reconstruct_batch(
                minibatch, sc_lambda=sc_lambda, step=step, **kwargs
            )

        if normed and out.any():
            with np.errstate(divide='ignore', invalid='ignore'):
                out /= np.max(np.abs(out), axis=1, keepdims=True)
            np.nan_to_num(out, copy=False)

        return out

    def reconstruct_auto(self, signal, *, zero_marg, lambda_lims, step=1, normed=True,
                         full_output=False, kwargs_bisect={}, kwargs_lasso={}):
        """TODO

        Reconstrueix un únic senyal buscant per bisecció la lambda que
        minimitza el senyal reconstruit al marge esquerre del senyal, la mida
        dels quals ve determinada per 'zero_marg'.

        """
        # Margins of the signals to be zeroed
        margin = signal[:zero_marg]
        # Function to be bisected.
        def fun(sc_lambda):
            rec, _ = self._reconstruct_single(margin, sc_lambda, step, **kwargs_lasso)
            return np.sum(np.abs(rec))

        try:
            with warnings.catch_warnings():
                # Ignore specific warning from extract_patches since here we do
                # not care about reconstructing the entire strain (margin).
                warnings.filterwarnings("ignore", message="'signals' cannot be fully divided into patches.*")
                result = lib.semibool_bisect(fun, *lambda_lims, **kwargs_bisect)
        
        except lib.BoundaryError:
            rec = np.zeros_like(signal)
            code = None
            result = {'x': np.min(lambda_lims), 'f': 0., 'converged': False, 'niters': 0, 'funcalls': 2}
        
        else:
            rec, code = self._reconstruct_single(signal, result['x'], step, **kwargs_lasso)
            if normed and rec.any():
                norm = np.max(np.abs(rec))
                rec /= norm
                code /= norm

        return (rec, code, result) if full_output else rec

    def reconstruct_iterative_minibatch(self, signals, sc_lambda=0.01, step=1, batchsize=64,
                                        max_iter=100, threshold=0.001, normed=True,
                                        full_output=False, verbose=True, kwargs_lasso={}):
        """Reconstruct multiple signals using iterative residual subtraction.

        This method reconstructs each signal by iteratively updating and
        accumulating reconstructions. In the first iteration, the original 
        input signal is reconstructed and then subtracted from itself to 
        obtain the initial residual. In each subsequent iteration, a new 
        reconstruction is generated from the current residual and subtracted 
        from it, producing an updated residual for the next iteration, while 
        also being added to the cumulative reconstruction. The process 
        repeats until the Euclidean norm of the difference between consecutive 
        residuals falls below a specified threshold, which sets the convergence 
        criterion.

        NOTE: In contrast with the usual procedure, the windows into which each
        signal is split are not normalized. This is needed to enhance the
        dictionary discrimination. Otherwise, the residuals are amplified at
        each iteration, the algorithm takes longer to converge, and some
        ad-hoc tests showed it also messes up with the resulting shape.

        Parameters
        ----------
        signals : ndarray
            Input signals to be reconstructed, with each signal along the first dimension.
        sc_lambda : float, optional
            Sparsity control parameter for reconstruction.
        step : int, optional
            Step size for the reconstruction.
        batchsize : int, optional
            Number of signals processed in each minibatch.
        max_iter : int, optional
            Maximum number of iterations before stopping.
        threshold : float, optional
            Convergence threshold based on the relative change in residuals.
        normed : bool, optional
            If True, the reconstructed signals are normalized after convergence.
        full_output : bool, optional
            If True, returns additional output values (residuals and iteration counts).
        verbose : bool, optional
            If True, prints progress information at each iteration.
        kwargs_lasso : dict, optional
            Additional arguments for the Lasso reconstruction method.

        Returns
        -------
        ndarray or tuple
            The final reconstructed signals. If `full_output` is True, also returns the residuals
            and the number of iterations per signal.

        """
        n_signals = signals.shape[0]

        # First iteration outside:
        if verbose:
                print(f"\nIteration 0")
                print(f"Signals remaining: {n_signals}")
        
        step_reconstructions = self.reconstruct_minibatch(
            signals, sc_lambda=sc_lambda, step=step, batchsize=batchsize,
            normed=False,  # Normalization is (optionally) applied at the END.
            normed_windows=False,  # See NOTE in the docstring.
            verbose=verbose, **kwargs_lasso
        )
        final_reconstructions = step_reconstructions.copy()
        residuals = signals - step_reconstructions

        # Stop conditions
        iters = np.ones(n_signals, dtype=int)
        finished = ~step_reconstructions.any(axis=1)  # In case any reconstructions are 0 already.
        residuals_old = residuals.copy()

        while not np.all(finished) and iters.max() < max_iter:
            if verbose:
                print(f"\nIteration {iters.max():3d}")
                print(f"Signals remaining: {(~finished).sum():^13d}")

            step_reconstructions = self.reconstruct_minibatch(
                residuals[~finished],
                sc_lambda=sc_lambda,
                step=step,
                batchsize=batchsize,
                normed=False,  # Normalization is (optionally) applied at the END.
                normed_windows=False,  # See NOTE in the docstring.
                verbose=verbose,
                **kwargs_lasso
            )
            final_reconstructions[~finished] += step_reconstructions
            residuals[~finished] -= step_reconstructions

            # Stop conditions
            iters[~finished] += 1
            residual_decrease = np.linalg.norm(residuals[~finished] - residuals_old[~finished], axis=1)
            finished[~finished] = residual_decrease < threshold
            residuals_old = residuals.copy()

            if verbose:
                print(
                    "CURRENT RESIDUAL DECREASE:\n"
                    f"Max: {residual_decrease.max()}\n"
                    f"Mean: {residual_decrease.mean()}\n"
                    f"Min: {residual_decrease.min()}\n"
                )

        if not np.all(finished):
            print("WARNING: reached max_iter before finishing all the reconstructions")

        if normed:
            with np.errstate(divide='ignore', invalid='ignore'):
                final_reconstructions /= np.max(np.abs(final_reconstructions), axis=1, keepdims=True)
            np.nan_to_num(final_reconstructions, copy=False)
        
        return (final_reconstructions, residuals, iters) if full_output else final_reconstructions


    def optimum_reconstruct(self, strain, *, reference, kwargs_minimize, kwargs_lasso,
                            step=1, limits=None, normed=True, verbose=False):
        """Find the best reconstruction of a signal w.r.t. a reference.

        Find the lambda which produces a reconstruction of the
        input 'strain' closest to the given 'reference', comparing them with
        the SSIM estimator. The search is performed by the SciPy's function
        'minimize_scalar' with bounds.

        PARAMETERS
        ----------
        strain: ndarray
            Input strain to be reconstructed (and optimized).

        reference: ndarray
            Reference strain which to compare the reconstruction to.

        kwargs_minimize: dict
            Passed to SciPy's `minimize_scalar(**kwargs_minimize)`.

        kwargs_lasso: dict
            Passed to Python-Spams' `lasso(**kwargs_lasso)`.

        step: int, optional
            Separation in samples between each window into which the input
            strain is split up to be reconstructed by the dictionary. Defaults
            to 1.

        limits: array-like, optional
            Indices of limits to where compute the loss between the
            reconstruction and the reference strain.

        normed: bool, optional
            If True, returns the signal normed to its maximum absolute amplitude.

        verbose: bool, optional
            Print info about the minimization results. False by default.

        RETURNS
        -------
        rec: ndarray
            Optimum reconstruction found.

        l_opt: float
            Optimum value for lambda.

        loss: float
            ISSIM (1 - SSIM) between the optimized reconstruction and the
            reference.

        """
        aa = 10
        bb = 10  # max(issim) x bb as the minimu value for the auxiliar line function.
        rec = None
        if limits is None:
            sl = slice(None)
        else:
            sl = slice(*limits)
        reference_ = reference[sl]

        def fun(l_rec_log):
            """Function to be minimized."""
            nonlocal rec
            l_rec = 10 ** l_rec_log  # Opitimizes lambda in log. space!
            rec = self.reconstruct(strain, l_rec, step=step, normed=normed, **kwargs_lasso)
            if rec.any():
                loss = estimators.issim(rec[sl], reference_)
            else:
                loss = aa * l_rec + bb
            return loss

        result = scipy.optimize.minimize_scalar(fun, **kwargs_minimize)
        l_opt = 10 ** result['x']
        loss = result['fun']

        if verbose:
            success = result['success']
            print(
                "Optimization results:\n"
                f"> Minimization success: {success}"
            )
            if not success:
                print(
                    "  Reason\n"
                    "  ------\n"
                    + result['message'] + "\n"
                    "  ------"
                )
            print(
                f"> Lambda optimized: {l_opt}\n"
                f"> Iterations performed: {result['nit']}\n"
                f"> Final loss: {loss}"
            )

        return rec, l_opt, loss

    def save(self, file):
        """Save the current state of the DictionarySpams object to a file.

        This method saves all attributes of the object as a `.npz` file. 
        If the object has not been trained, certain attributes (`lambda1`, 
        `n_train`, and `t_train`) are removed to avoid potential issues when 
        reloading the state.

        Parameters
        ----------
        file : str or file-like object
            The file path or file object where the state of the object will 
            be saved. If a string is provided, it specifies the path to the 
            `.npz` file. If a file-like object is given, it must be writable 
            in binary mode.

        """
        vars_ = vars(self)
        to_remove = []

        if not self.trained:
            # To avoid silent bugs in the future
            to_remove += ['lambda1', 'n_train', 't_train']

        for attr in to_remove:
            vars_.pop(attr)

        np.savez(file, **vars_)

    def copy(self):
        """Return a copy of the dictionary.

        Returns a new instance of the same dictionary with the same values
        and state.

        Returns
        -------
        dico_copy : DictionarySpams
            A copy of the current dictionary.
        
        """
        dico_copy = DictionarySpams(
            dict_init=self.components.copy(),
            model=self.model,
            lambda1=self.lambda1,
            batch_size=self.batch_size,
            identifier=self.identifier,
            n_iter=self.n_iter,
            n_train=self.t_train,
            trained=self.trained,
            mode_traindl=self.mode_traindl,
            modeD_traindl=self.modeD_traindl,
            mode_lasso=self.mode_lasso
        )
        if self.trained:
            # Retain the initial components of the dictionary.
            dico_copy.dict_init = self.dict_init

        return dico_copy
    
    def reset(self):
        """Reset the dictionary to its initial (untrained) state."""
        self.components = self.dict_init
        self.trained = False
        self.n_train = None
        self.t_train = None

    def _check_initial_parameters(self, signal_pool, ignore_completeness):
        # Explicit initial dictionary.
        if self.dict_init is not None:
            if not isinstance(self.dict_init, np.ndarray):
                raise TypeError(
                    f"'{type(self.dict_init).__name__}' is not a valid 'dict_init'"
                )
            
            if not self.dict_init.flags.c_contiguous:
                raise ValueError("'dict_init' must be a C-contiguous array")
            
            if (self.dict_init.shape[1] >= self.dict_init.shape[0]
                    and not ignore_completeness):
                raise ValueError("the dictionary must be overcomplete (d_size > a_length)")
        
        # Signal pool from where to extract the initial dictionary.
        elif signal_pool is not None:
            if not isinstance(signal_pool, np.ndarray):
                raise TypeError(
                    f"'{type(signal_pool).__name__}' is not a valid 'signal_pool'"
                )
            
            if not signal_pool.flags.c_contiguous:
                raise ValueError("'signal_pool' must be a C-contiguous array")
            
            if None in (self.a_length, self.d_size):
                raise TypeError(
                    f"'a_length' and 'd_size' must be explicitly provided along 'signal_pool'"
                )
            
            if (self.a_length >= self.d_size) and not ignore_completeness:
                raise ValueError("the dictionary must be overcomplete (d_size > a_length)")
        
        else:
            raise ValueError("either 'dict_init' or 'signal_pool' must be provided")
