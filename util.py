import warnings

import numpy as np


_xtol = 2e-12
_rtol = 8.881784197001252e-16


class BoundaryError(ValueError):
    pass


def abs_normalize(array, axis=0):
    """TODO
    Normalitza inplace un array ignorant els errors de divissió entre 0 i
    canviant els nan a 0.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        array /= np.max(np.abs(array), axis=axis, keepdims=True)
        np.nan_to_num(array, copy=False)


def l2_normalize(array, axis=0):
    """TODO
    Normalitza inplace un array amb la norma L2 ignorant els errors de divissió
    entre 0 i canviant els nan a 0.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        array /= np.linalg.norm(array, axis=axis, keepdims=True)
        np.nan_to_num(array, copy=False)


def semibool_bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=100, verbose=False):
    """TODO

    Troba x0 pel mètode de bisecció adaptat a una funció f(x) tal que
        f(x)  > 0       x <= x0,
        f(x) == 0       x  > x0,
    o viceversa. Un dels dos extrems del límite [a, b] ha de ser f(x) = 0.
    Algorisme basat en la funció de bisecció `scipy.optimize.bisect`.

    Nota: 'rtol' controla l'ordre de precissió respecte 'x'.

    Result
    ------
    solver_stats: dict
        'x': Solution.
        'f': Value of f(x).
        'converged': bool.
        'niters': Number of iterations performed.
        'funcalls': Number of times `f` was evaluated.

    """
    fa = f(a, *args)
    fb = f(b, *args)
    solver_stats = {'funcalls': 2}
    if (fa*fb != 0) or (fa == fb == 0):
        raise BoundaryError("There isn't a boundary point in the 0 zone")
    if fa == 0:
        a, b = b, a
        fa, fb = fb, fa
    
    dm = b - a
    for i in range(maxiter):
        dm *= 0.5
        xm = a + dm
        if verbose:
            print(f" iteration {i}, evaluating f({xm}) ...")
        fm = f(xm, *args)
        solver_stats['funcalls'] += 1
        if fm != 0:
            a = xm
            fa = fm
        if abs(dm) < xtol + rtol*abs(xm):
            solver_stats['converged'] = True
            solver_stats['niters'] = i+1
            solver_stats['x'] = a  # Last point where f(x) != 0
            solver_stats['f'] = fa
            return solver_stats

    # Not converged
    solver_stats['converged'] = False
    solver_stats['niters'] = i+1
    solver_stats['x'] = a
    solver_stats['f'] = fa
    
    return solver_stats


def extract_patches(signals, *, patch_size, n_patches=None, random_seed=None,
                    step=1, limits=None, patch_min=1, l2_normed=False, return_norm_coefs=False,
                    allow_allzeros=True):
    """Extract patches (all possible or randomly selected) from the input 'signals'.

    Note that if randomly selected, it is not prevented to repeat patches.

    PARAMETERS
    ----------
    signals: ndarray
        If 2d-array must be in fortran order with shape (l_signal, n_signals).

    patch_size: int
        Length of the patches to extract.

    limits: ndarray, optional
        Limit(s) in 'signals' such that:
            ```
            i0, i1 = limits[i_signal]
            signal = signals[i0:i1,i_signal]
            ```.

    patch_min: int, optional
        When 'limits' are given, patch_min is the minimum samples inside the limits to be
        included in the extracted patches. Defaults to 1.

    n_patches: int, optional
        Total number of patches to extract. If None (default) extract the maximum amount.

    random_seed: {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        Given to 'numpy.random.default_rng(random_seed)'.

    step: int, optional
        Minimum windowing step (separation between patches).

    l2_normed: bool, optional
        If True will norm each patch to its L2 norm. False by default.

    return_norm_coefs: bool, optional
        If True, return also the coefficients used to normalize the signals (useful for when the
        'signals' are from a windowed signal that will be reassembled afterwards).
        False by default.

    allow_allzeros: bool, optional
        When extracting random patches, if False and l2_normed=True,
        generate another random window position until the l2 norm is != 0.
    
    """
    if signals.ndim > 2:
        raise ValueError("'signals' must be 2d-array at most")
    if signals.ndim == 1:
        signals = signals[:,None]

    # Compute the maximum patches per signal that can be obtained with the given 'step'
    # ignoring the limits:
    rng = np.random.default_rng(random_seed)
    l_signals, n_signals = signals.shape
    max_pps = (l_signals - patch_size) / step + 1
    if not max_pps.is_integer() and limits is None and signals.ndim == 1:
        warnings.warn(
            "'signals' cannot be fully divided into patches, the last"
            f" {(max_pps-1)*step % step:.0f} bins of each signal will be left out",
            RuntimeWarning
        )
    max_pps = int(max_pps)

    # Compute the maximum TOTAL number of patches and the limits from where to extract
    # patches for each signal.
    if limits is None:
        window_limits = [(0, l_signals-patch_size+1)] * n_signals
        max_patches = max_pps * n_signals
    else:
        window_limits = []
        max_patches = 0
        for p0, p1 in limits:
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
    else:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            signal_aux = signals[j:j+patch_size,i]
            if not allow_allzeros:
                while not signal_aux.any():
                    j = rng.integers(*window_limits[i])
                    signal_aux = signals[j:j+patch_size,i]
            patches[:,k] = signal_aux

    # Normalize each patch to its L2 norm
    if l2_normed:
        coefs = np.linalg.norm(patches, axis=0)
        # Ignore x/0 and 0/0 cases
        with np.errstate(divide='ignore', invalid='ignore'):
            patches /= coefs
        patches = np.nan_to_num(patches)

    return (patches, coefs) if return_norm_coefs else patches


def reconstruct_from_patches_1d(patches, step):
    l_patches, n_patches = patches.shape
    total_len = (n_patches - 1) * step + l_patches
    
    reconstructed = np.zeros(total_len)
    normalizer = np.zeros_like(reconstructed)
    for i in range(n_patches):
        reconstructed[i*step:i*step+l_patches] += patches[:,i]
        normalizer[i*step:i*step+l_patches] += 1
    normalizer[i*step+l_patches:] = 1
    reconstructed /= normalizer

    return reconstructed

FINISHED = f"\n\n{''.join(chr(x) for x in (9995,128524,128076))}"


def inject(strain, *, background, snr, psd, at, limits=None, offset=0, window=('tukey',0.5)):
    """Inject a strain into a background strain.

    PARAMETERS
    ----------
    strain: array
        Strain to be injected.

    background: array
        Background strain where to inject the `strain` to.
        Should be `len(background) => len(strain)`.

    limits: array-like (size=2), optional
        Initial and final position of 'strain'. Useful when the actual signal
        is zero-padded.
    
    offset: int
        Index position in 'background' where to begin the injection.
    
    **kwargs: passed to snr()

    RETURNS
    -------
    injected: array
        Injected strain with `len(injected) == len(background)`.

    """
    if limits is None:
        strain_ = strain_
    else:
        strain_ = strain[slice(*limits)]
    snr0 = compute_snr(strain_, psd=psd, at=at, window=window)
    i0 = offset
    i1 = i0 + len(strain_)
    injected = background.copy()
    injected[i0:i1] += strain_ * snr/snr0

    return injected


def inject_batch(strain_set, *, background, snr, psd, at, limits=None, offset=0,
                 window=('tukey',0.5)):
    """Inject a batch of strains into a background strain.

    PARAMETERS
    ----------
    strain_set: 2d-array (#-strains, length)
        Set of strains to be injected.

    background: array
        First or background strain where to inject the `inject_strain` to.
        Should be `len(background) => strain_set.shape[1]`.

    limits: 2d-array (#-strains, 2), optional
        Initial and final position of each strain in 'strain_set'. Useful when
        the actual signals are zero-padded.
    
    offset: int
        Index position in 'background' where to begin the injection.
        By default the injection is performed at the beginning.
    
    **kwargs: passed to snr()

    RETURNS
    -------
    set_injected: 2d-array (length, #-strains)
        Injected strains with `injected.shape == background.shape`.

    """
    l_s = len(background)  # length of the final strains
    n_s = len(strain_set)  # number of strains to inject
    if limits is None:
        limits = [None] * n_s

    set_injected = np.empty((n_s, l_s))
    for i in range(n_s):
        to_inject = strain_set[i]
        set_injected[i] = inject(
            to_inject, background=background, snr=snr, psd=psd, at=at, limits=limits[i],
            offset=offset, window=window
        )

    return set_injected
