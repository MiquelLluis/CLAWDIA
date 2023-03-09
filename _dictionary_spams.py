import time

import numpy as np
import spams

from ._dictionary_base import _DictionaryBase
from . import util


# Remove warning from OpenMP, present in older versions of python-spams.
if not '__version__' in dir(spams) or spams.__version__ <= '2.6.5.4':
    import os
    os.environ['KMP_WARNINGS'] = 'FALSE'


class DictionarySpams(_DictionaryBase):
    """Mini-Batch Dictionary Learning interface for SPAMS-python.

    Set of utilities for dictionary learning and sparse encoding using the
    functions of SPAMS-python[1].

    Parameters
    ----------
    dict_init : 2d-array(p_size, d_size), optional
        Atoms of the initial dictionary.
        If None, 'signal_pool' must be given.

    signal_pool : 2d-array(samples, signals), optional
        Set of signals from where to randomly extract the atoms.
        Ignored if 'dict_init' is not None.

    wave_pos : 2d array-like (len(signals), 2), optional
        Position of each waveform inside 'signal_pool' from where to extract
        the atoms for the initial dictionary.
        If None, the whole array will be used.

    p_size : int, optional
        Atom length (patch size).
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
        Kwarg to pass to util.extract_patches if initializing the
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

    mode_traindl : int, 0 by default
        Refer to [1] for more information.

    mode_lasso : int, 2 by default
        Refer to [1] for more information.

    Attributes
    ----------
    dict_init : array(p_size, d_size)
        Atoms of the initial dictionary.

    components : array(p_size, d_size)
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
    def __init__(self, dict_init=None, signal_pool=None, wave_pos=None, p_size=None, d_size=None,
                 lambda1=None, batch_size=64, identifier='', l2_normed=True, allow_allzeros=True,
                 n_iter=None, n_train=None, patch_min=1, random_state=None, trained=False,
                 ignore_completeness=False, mode_traindl=0, mode_lasso=2):
        self.dict_init = dict_init
        self.components = dict_init
        self.wave_pos = wave_pos
        self.p_size = p_size
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
        self.mode_lasso = mode_lasso

        self._check_initial_parameters(signal_pool)

        # Explicit initial dictionary (trained or not).
        if self.dict_init is not None:
            self.p_size, self.d_size = self.dict_init.shape

        # Get the initial atoms from a set of signals.
        else:
            self.dict_init = util.extract_patches(
                signal_pool,
                self.p_size,
                limits=self.wave_pos,
                n_patches=self.d_size,
                l2_normed=self.l2_normed,
                allow_allzeros=self.allow_allzeros,
                patch_min=self.patch_min,
                random_state=self.random_state
            )
            self.components = self.dict_init

    def train(self, patches, lambda1=None, n_iter=None, verbose=False, **kwargs):
        """Train the dictionary with a set of patches.

        Calls 'spams.trainDL' to train the dictionary by solving the
        learning problem
            min_{D in C} (1/d_size) sum_{i=1}^d_size {
                (1/2)||x_i-Dalpha_i||_2^2  s.t. ||alpha_i||_1 <= lambda1
            } .

        Parameters
        ----------
        patches : 2d-array(samples, signals)
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

        **kwargs
            Passed directly to 'spams.trainDL', see [1].

        Additional parameters will be passed to the SPAMS training function.

        """
        if len(patches) != self.p_size:
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

        self.n_train = patches.shape[1]

        tic = time.time()
        self.components, model = spams.trainDL(
            patches,
            D=self.dict_init,  # Cool-start
            batchsize=self.batch_size,
            lambda1=self.lambda1,
            iter=self.n_iter,
            mode=self.mode_traindl,  # Default mode is 2
            verbose=verbose,
            return_model=True,
            **kwargs
        )
        tac = time.time()

        self.trained = True

        if self.n_iter < 0:
            self.t_train = -self.n_iter
            self.n_iter = model['iter']
        else:
            self.t_train = tac - tic

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

    def _reconstruct(self, signal, sc_lambda, step=1, **kwargs):
        patches, norms = util.extract_patches(
            signal,
            patch_size=self.p_size,
            step=step,
            l2_normed=True,
            return_norm_coefs=True
        )
        code = spams.lasso(
            patches,
            D=self.components,
            lambda1=sc_lambda,
            mode=self.mode_lasso,
            **kwargs
        )
        patches = (self.components @ code) * norms

        signal_rec = util.reconstruct_from_patches_1d(patches, step)

        return signal_rec, code

    def _check_initial_parameters(self, signal_pool):
        # Explicit initial dictionary.
        if self.dict_init is not None:
            if not isinstance(self.dict_init, np.ndarray):
                raise TypeError(
                    f"'{type(self.dict_init).__name__}' is not a valid 'dict_init'"
                )
            if not self.dict_init.flags.f_contiguous:
                raise ValueError("'dict_init' must be a F-contiguous array")
            if (self.dict_init.shape[0] >= self.dict_init.shape[1]
                and not self.ignore_completeness):
                raise ValueError("the dictionary must be overcomplete (p_size < d_size)")
        
        # Signal pool from where to extract the initial dictionary.
        elif signal_pool is not None:
            if not isinstance(signal_pool, np.ndarray):
                raise TypeError(
                    f"'{type(signal_pool).__name__}' is not a valid 'signal_pool'"
                )
            if not signal_pool.flags.f_contiguous:
                raise ValueError("'signal_pool' must be a F-contiguous array")
            if None in (self.p_size, self.d_size):
                raise TypeError(
                    f"'p_size' and 'd_size' must be explicitly provided along 'signal_pool'"
                )
            if self.p_size >= self.d_size:
                raise ValueError("the dictionary must be overcomplete (p_size < d_size)")
        
        # None of the above.
        else:
            raise ValueError("either 'dict_init' or 'signal_pool' must be provided")
