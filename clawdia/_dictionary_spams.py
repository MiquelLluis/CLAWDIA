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
    """Mini-Batch Dictionary Learning interface for SPAMS-python.

    Set of utilities for dictionary learning and sparse encoding using the
    functions of SPAMS-python[1].

    Parameters
    ----------
    dict_init : 2d-array(d_size, a_length), optional
        Atoms of the initial dictionary.
        If None, 'signal_pool' must be given.

    signal_pool : 2d-array(signals, samples), optional
        Set of signals from where to randomly extract the atoms.
        Ignored if 'dict_init' is not None.

    wave_pos : 2d array-like (len(signals), 2), optional
        Position of each waveform inside 'signal_pool' from where to extract
        the atoms for the initial dictionary.
        If None, the whole array will be used.

    a_length : int, optional
        Atoms' length (patch size).
        If 'signal_pool' is not None, must be given.

    d_size : int, optional
        Number of atoms (dictionary size).
        If 'signal_pool' is not None, must be given.

    lambda1 : float, optional
        Regularization parameter of the learning algorithm.
        If None, will be requierd when calling 'train' method.

    batch_size : int, 64 by default
        Number of samples in each mini-batch.

    identifier : str, optional
        A word or note to identify the dictionary.

    l2_normed : bool, True by default
        If True, normalize atoms to their L2-Norm.

    allow_allzeros : bool, True by default
        Kwarg to pass to lib.extract_patches if initializing the
        dictionary from a signal_pool.

    n_iter : int, optional
        Total number of iterations to perform.
        If a negative number is provided it will perform the computation during
        the corresponding number of seconds. For instance n_iter=-5 learns the
        dictionary during 5 seconds.
        If None, will be required when calling 'train' method.

    n_train : int, optional
        Number of patches used to train the dictionary in case it has been
        trained already (just informative).

    patch_min : int, 1 by default
        Minimum number of samples within each 'wave_pos[i]' to include in each
        extracted atom when 'signal_pool' given.
        Will be ignored if 'wave_pos' is None.

    random_state : int, optional
        Seed used for random sampling.

    trained : bool, False by default
        Flag indicating whether dict_init is an already trained dictionary.

    ignore_completeness : bool, optional
        If set to True and the dictionary is not overcomplete, no error will be
        raised.

    mode_traindl : int, 0 by default
        Refer to [1] for more information.
    
    modeD_traindl : int, 0 by default
        Refer to [1] for more information.

    mode_lasso : int, 2 by default
        Refer to [1] for more information.

    Attributes
    ----------
    dict_init : array(d_size, a_length)
        Atoms of the initial dictionary.

    components : array(d_size, a_length)
        Atoms of the current dictionary.

    n_iter : int
        Number of iterations performed in training.

    t_train : float
        Time spent training.

    identifier : str
        A word or note to identify the dictionary.

    References
    ----------
    [1]: SPAMS (for python), (http://spams-devel.gforge.inria.fr/), last
    accessed in october 2018.

    [2]: SciPy's Optimization tools,
    (https://docs.scipy.org/doc/scipy/reference/optimize.html), last accessed
    in February 2022.

    """
    def __init__(self, dict_init=None, signal_pool=None, wave_pos=None, a_length=None, d_size=None,
                 lambda1=None, batch_size=64, identifier='', l2_normed=True, allow_allzeros=True,
                 n_iter=None, n_train=None, patch_min=1, random_state=None, trained=False,
                 ignore_completeness=False, mode_traindl=0, modeD_traindl=0, mode_lasso=2):
        self.dict_init = dict_init
        self.components = dict_init
        self.wave_pos = wave_pos
        self.a_length = a_length
        self.d_size = d_size
        self.lambda1 = lambda1
        self.batch_size = batch_size
        self.identifier = identifier
        self.l2_normed = l2_normed
        self.allow_allzeros = allow_allzeros
        self.n_iter = n_iter
        self.t_train = -n_iter if n_iter is not None and n_iter < 0 else None
        self.n_train = n_train
        self.patch_min = patch_min
        self.random_state = random_state
        self.trained = trained
        self.ignore_completeness = ignore_completeness
        self.mode_traindl = mode_traindl
        self.modeD_traindl = modeD_traindl
        self.mode_lasso = mode_lasso

        self._check_initial_parameters(signal_pool)

        # Explicit initial dictionary (trained or not).
        if self.dict_init is not None:
            self.d_size, self.a_length = self.dict_init.shape

        # Get the initial atoms from a set of signals.
        else:
            self.dict_init = lib.extract_patches(
                signal_pool,
                patch_size=self.a_length,
                limits=self.wave_pos,
                n_patches=self.d_size,
                l2_normed=self.l2_normed,
                allow_allzeros=self.allow_allzeros,
                patch_min=self.patch_min,
                random_state=self.random_state
            )
            self.components = self.dict_init

    def train(self, patches, lambda1=None, n_iter=None, verbose=False, threads=-1, **kwargs):
        """Train the dictionary with a set of patches.

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
            It is not needed if already specified at initialization.

        verbose : bool, optional
            If True print the iterations (might not be shown in real time).

        threads : int, optional
            Number of threads to use during training, see [1].

        **kwargs
            Passed directly to 'spams.trainDL', see [1].

        Additional parameters will be passed to the SPAMS training function.

        """
        if patches.shape[1] != self.a_length:
            raise ValueError("the length of 'patches' must be the same as the"
                             " atoms of the dictionary")
        if n_iter is not None:
            self.n_iter = n_iter
        elif self.n_iter is None:
            raise TypeError("'n_iter' not specified")
            
        if lambda1 is not None:
            self.lambda1 = lambda1
        elif self.lambda1 is None:
            raise TypeError("'lambda1' not specified")

        self.n_train = patches.shape[0]

        # In case of loading older instances in which this attribute didn't
        # exist, it is set to the default of spams.trainDL. This way it should
        # produce the same results as before.
        if not hasattr(self, 'modeD_traindl'):
            self.modeD_traindl = 0

        tic = time.time()
        components, model = spams.trainDL(
            patches.T,           # SPAMS works with Fortran order.
            D=self.dict_init.T,  #
            batchsize=self.batch_size,
            lambda1=self.lambda1,
            iter=self.n_iter,
            mode=self.mode_traindl,
            modeD=self.modeD_traindl,
            verbose=verbose,
            numThreads=threads,
            return_model=True,
            **kwargs
        )
        self.components = components.T
        tac = time.time()

        self.trained = True

        if self.n_iter < 0:
            self.t_train = -self.n_iter
            self.n_iter = model['iter']
        else:
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
        """Reconstruct multiple signals by iterative subtraction.

        Each signal is reconstructed by iteratively performing a reconstruction
        of the input signal, subtracting it to the original, and then
        performing again the reconstruction on the residual until the relative
        difference between consecutive residuals is below a certain threshold.

        NOTE: During the step reconstructions, the windows into which each
        signal is split are not normalized. This is needed to enhance the
        dictionary discrimination. Otherwise, the residuals are amplified at
        each iteration, the algorithm takes longer to converge, and some
        ad-hoc tests showed it also messes up with the resulting shape.

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
        vars_ = vars(self)
        to_remove = []

        if not self.trained:
            # To avoid silent bugs in the future
            to_remove += ['lambda1', 'n_train', 't_train']
        if self.wave_pos is None:
            to_remove.append('wave_pos')
        if self.random_state is None:
            to_remove.append('random_state')

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
            wave_pos=self.wave_pos.copy(),
            lambda1=self.lambda1,
            batch_size=self.batch_size,
            identifier=self.identifier,
            l2_normed=self.l2_normed,
            allow_allzeros=self.allow_allzeros,
            n_iter=self.n_iter,
            n_train=self.t_train,
            patch_min=self.patch_min,
            random_state=self.random_state,
            trained=self.trained,
            ignore_completeness=self.ignore_completeness,
            mode_traindl=self.mode_traindl,
            modeD_traindl=self.modeD_traindl,
            mode_lasso=self.mode_lasso
        )
        if self.trained:
            # Retain the initial components of the dictionary.
            dico_copy.dict_init = self.dict_init

        return dico_copy

    def _check_initial_parameters(self, signal_pool):
        # Explicit initial dictionary.
        if self.dict_init is not None:
            if not isinstance(self.dict_init, np.ndarray):
                raise TypeError(
                    f"'{type(self.dict_init).__name__}' is not a valid 'dict_init'"
                )
            
            if not self.dict_init.flags.c_contiguous:
                raise ValueError("'dict_init' must be a C-contiguous array")
            
            if (self.dict_init.shape[1] >= self.dict_init.shape[0]
                    and not self.ignore_completeness):
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
            
            if (self.a_length >= self.d_size) and not self.ignore_completeness:
                raise ValueError("the dictionary must be overcomplete (d_size > a_length)")
        
        else:
            raise ValueError("either 'dict_init' or 'signal_pool' must be provided")
