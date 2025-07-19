"""Collection of auxiliary functions.

This module provides a temporary collection of various utility functions, 
including mathematical operations, signal normalization, optimization routines, 
and signal patching utilities.

Notes
-----
As the CLAWDIA pipeline grows, these functions will be organized into more
semantically appropriate modules.

"""
import warnings

import numpy as np


# Default values same as Scipy's `optimize._zeros_py.py` module.
# Ref: https://github.com/scipy/scipy/blob/v1.15.0/scipy/optimize/_zeros_py.py
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps


class BoundaryError(ValueError):
    """Exception raised when a boundary condition is violated.
    
    This exception is a subclass of `ValueError` and is used to signal
    issues with input values exceeding or falling below defined boundaries.
    
    """
    pass


def abs_normalize(array, axis=-1):
    """TODO
    Normalitza inplace un array ignorant els errors de divissió entre 0 i
    canviant els nan a 0.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        array /= np.max(np.abs(array), axis=axis, keepdims=True)
        np.nan_to_num(array, copy=False)


def l2_normalize(array, axis=-1):
    """TODO
    Normalitza inplace un array amb la norma L2 ignorant els errors de divissió
    entre 0 i canviant els nan a 0.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        array /= np.linalg.norm(array, axis=axis, keepdims=True)
        np.nan_to_num(array, copy=False)


def semibool_bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=100, verbose=False):
    """Semi-boolean bisection method for solving ``f(x)``.

    Find the boundary point ``x0`` in the interval [a, b] using a modified
    bisection method such that:

    - ``f(x) > 0 for x <= x0``
    - ``f(x) == 0 for x > x0``

    or vice versa. One of the two endpoints of the interval [a, b] must
    satisfy ``f(x) = 0``.
    The method iteratively narrows down the interval [a, b] until the
    solution is found to within the specified tolerances `xtol` and `rtol`, or
    the maximum number of iterations is reached.
    
    This algorithm is based on the `scipy.optimize.bisect` method, but includes
    modifications for this specific boundary behavior.

    Parameters
    ----------
    f : callable
        A continuous function of a single variable, ``f(x)``. The function must
        accept a single positional argument and any additional arguments via
        `args`.
    a : float
        Lower bound of the interval [a, b].
    b : float
        Upper bound of the interval [a, b].
    args : tuple, optional
        Additional arguments to pass to the function `f`.
    xtol : float, optional
        Absolute tolerance for the solution. Default is ``2e-12``.
    rtol : float, optional
        Relative tolerance for the solution. Default is `_rtol`.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is 100.
    verbose : bool, optional
        If `True`, prints the progress of each iteration, including the
        evaluation of ``f(x)`` at the midpoint. Default is `False`.

    Returns
    -------
    solver_stats : dict
        A dictionary containing information about the solution:
        
        - `x` : float
           The last point where ``f(x) != 0``. Approximates the boundary point
           `x0`.
        - `f` : float
           The value of ``f(x)`` at the solution.
        - `converged` : bool
           Indicates whether the algorithm converged to a solution.
        - `niters` : int
           The number of iterations performed.
        - `funcalls` : int
           The total number of function evaluations.

    Raises
    ------
    BoundaryError
        If the initial interval [a, b] does not satisfy the condition that
        one endpoint evaluates to ``f(x) = 0``.
    ValueError
        If the function is not continuous or if `a` and `b` are not proper
        bounds.

    Notes
    -----
    - The `xtol` parameter controls the absolute precision of the solution,
      while the `rtol` parameter ensures relative precision with respect to the
      value of `x`.
    - This method assumes that ``f(x)`` contains exactly one boundary point
      within [a, b].

    Examples
    --------
    >>> def f(x):
    ...     return 0 if x < 2 else x
    >>> result = semibool_bisect(f, 0, 4)
    >>> print(result['x'])
    2.0
    >>> print(result['converged'])
    True

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


def extract_patches(signals, *, patch_size, n_patches=None, random_state=None,
                    step=1, limits=None, patch_min=1, l2_normed=False,
                    return_norm_coefs=False, allow_allzeros=True):
    """Extract patches from 'signals'.

    TODO

    Note that if randomly selected, it is not prevented to repeat patches.

    PARAMETERS
    ----------
    signals: ndarray
        If 2d-array, must be in C-contiguous order with shape
        (n_signals, l_signal).

    patch_size: int
        Length of the patches to extract.

    limits: ndarray, optional
        Limit(s) in 'signals' such that:
            
            ```
            i0, i1 = limits[i_signal]
            signal = signals[i_signal, i0:i1]
            ```

    patch_min: int, optional
        When 'limits' are given, patch_min is the minimum samples inside the
        limits to be included in the extracted patches.
        Defaults to 1.

    n_patches: int, optional
        Total number of patches to extract. If None (default) extract the
        maximum amount.

    random_state: {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        Given to 'numpy.random.PCG64(random_state)'.

    step: int, optional
        Minimum windowing step (separation between patches).

    l2_normed: bool, optional
        If True will norm each patch to its L2 norm. False by default.

    return_norm_coefs: bool, optional
        If True, return also the coefficients used to normalize the signals
        (useful for when 'signals' come from a windowed signal that will be
        reassembled afterwards).
        False by default.

    allow_allzeros: bool, optional
        When extracting random patches, if False and `l2_normed == True`,
        generate another random window position until the l2 norm is != 0.
    
    """
    if signals.ndim > 2:
        raise ValueError("'signals' must be 2d-array at most")
    if limits is not None:
        if limits.shape[1] != 2:
            raise ValueError(
                f"'limits' has a wrong shape: {limits.shape}"
            )
        if patch_min > np.min(np.diff(limits, axis=1)):
            raise ValueError(
                "there is at least one signal according to its 'limits' shorter"
                " than 'patch_min'"
            )
    if not allow_allzeros and not signals.any():
        raise ValueError(
            "'allow_allzeros' is False, but 'signals' contains only zeros. "
            "Random patch extraction would result in an infinite loop."
        )
    
    if signals.ndim == 1:
        signals = signals[np.newaxis,:]

    

    # Compute the maximum patches per signal that can be obtained with the
    # given 'step' ignoring the limits.

    rng = np.random.Generator(np.random.PCG64(random_state))
    n_signals, l_signals = signals.shape
    max_pps = (l_signals - patch_size) / step + 1
    if not max_pps.is_integer() and limits is None and n_signals == 1:
        warnings.warn(
            "'signals' cannot be fully divided into patches, the last"
            f" {(max_pps-1)*step % step:.0f} bins of each signal will be left out",
            RuntimeWarning
        )
    max_pps = int(max_pps)


    
    # Compute the maximum TOTAL number of patches and the limits from where to
    # extract patches for each signal.

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
    
    
    
    patches = np.empty((n_patches, patch_size), dtype=signals.dtype)

    # Extract all possible patches.
    if n_patches == max_patches:
        k = 0
        for i in range(n_signals):
            p0, p1 = window_limits[i]
            for j in range(p0, p1, step):
                patches[k] = signals[i, j:j+patch_size]
                k += 1
    
    # Extract a limited number of patches randomly selected.
    else:
        for k in range(n_patches):
            i = rng.integers(0, n_signals)
            j = rng.integers(*window_limits[i])
            signal_aux = signals[i, j:j+patch_size]
            if not allow_allzeros:
                while not signal_aux.any():
                    j = rng.integers(*window_limits[i])
                    signal_aux = signals[i, j:j+patch_size]
            patches[k] = signal_aux

    # Normalize each patch to its L2 norm
    if l2_normed:
        coefs = np.linalg.norm(patches, axis=1, keepdims=True)
        # Ignore x/0 and 0/0 cases
        with np.errstate(divide='ignore', invalid='ignore'):
            patches /= coefs
        patches = np.nan_to_num(patches)

        coefs = np.ravel(coefs)
    
    elif return_norm_coefs:
        coefs = np.ones(n_patches)

    return (patches, coefs) if return_norm_coefs else patches


def reconstruct_from_patches_1d(patches, step):
    n_patches, l_patches = patches.shape
    total_len = (n_patches - 1) * step + l_patches
    
    reconstructed = np.zeros(total_len, dtype=patches.dtype)
    normalizer = np.zeros_like(reconstructed)
    for i in range(n_patches):
        reconstructed[i*step:i*step+l_patches] += patches[i]
        normalizer[i*step:i*step+l_patches] += 1
    normalizer[i*step+l_patches:] = 1
    reconstructed /= normalizer

    return reconstructed


FINISHED = "FINISHED " + f"\n\n{''.join(chr(x) for x in (9995,128524,128076))}"
