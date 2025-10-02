"""
Estimators and metrics for signal analysis and comparison.

This module provides a variety of functions to compute statistical and signal-processing 
metrics, such as mean squared error, structural similarity index, overlaps, 
signal-to-noise ratios, and others. While some functions are specifically designed for 
gravitational-wave signal analysis, they can also be applied to broader signal-processing 
contexts.

"""
import numpy as np
import scipy as sp



def mse(x, y):
    """Mean Squared Error."""
    return float(np.mean((x-y)**2) / len(x))


def medse(x, y):
    """Median Squared Error."""
    return float(np.median((x-y)**2))


def ssim(x, y):
    """Structural Similarity Index Measure (SSIM).

    Compute the Structural Similarity Index Measure (SSIM) between two
    arrays, `x` and `y`. SSIM is a perceptual metric that quantifies the
    similarity between two signals or images, accounting for luminance,
    contrast, and structure [1]_, [2]_.

    Reference values:

         1 → Perfect similarity.
         0 → No similarity.
        -1 → Perfect anti-correlation.

    Parameters
    ----------
    x : array_like
        Input signal or image. Must be of the same shape as `y`.
    y : array_like
        Input signal or image. Must be of the same shape as `x`.

    Returns
    -------
    res : float
        The Structural Similarity Index Measure between the signals `x` and `y`.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
        Image quality assessment: From error visibility to structural similarity. 
        IEEE Transactions on Image Processing, 13(4), 600-612.
    .. [2] https://en.wikipedia.org/wiki/Structural_similarity
    
    """
    mux = x.mean()
    muy = y.mean()
    cov_mat = np.cov(x, y, ddof=0)
    sx2 = cov_mat[0, 0]
    sy2 = cov_mat[1, 1]
    sxy = cov_mat[0, 1]
    l_ = 1
    c1 = (0.01*l_) ** 2
    c2 = (0.03*l_) ** 2

    res = float(
        (2 * mux * muy + c1) * (2 * sxy + c2)
        / ((mux**2 + muy**2 + c1) * (sx2 + sy2 + c2))
    )
    
    return res


def dssim(x, y):
    """Structural Dissimilarity.
    
    Reference values:

        0 → Perfect correlation.
        ½ → No correlation.
        1 → Perfect anticorrelation.
    """
    return (1 - ssim(x, y)) / 2


def issim(x, y):
    """Inverse Structural SimilarityIndex Measure.
    
    In this case:

        -  1: Perfect anti-correlation.

        -  0: No similarity.

        - -1: Perfect similarity.

    Useful as a loss function to perform minimization.
        
    """
    return -ssim(x, y)


def residual(x, y):
    """Norm of the difference between 'x' and 'y'."""
    return np.linalg.norm(x - y)


def softmax(x, axis=None):
    """Softmax probability distribution."""
    coefs = np.exp(x)
    return coefs / coefs.sum(axis=axis, keepdims=True)


def inner_product_weighted(x, y, *, at, psd=None, window='hann'):
    """Compute the weighted inner product (x|y) between two signals.
    
    Parameters
    ----------
    x, y: ndarray
        Signals to compare.

    at: float
        Sample time step.
    
    psd: 2d-array, optional
        PSD to weight the overlap, will be linearly interpolated to the right frequencies.
        psd[0] = frequencies
        psd[1] = psd samples
    
    References
    ----------
    [1]: Eq. 12, DOI: 10.48550/arxiv.2210.06194
    
    """    
    ns = len(x)
    if ns != len(y):
        raise ValueError("both 'x' and 'y' must be of the same length")
    if not np.isrealobj(x):
        raise ValueError(f"'x' cannot be complex")
    if not np.isrealobj(y):
        raise ValueError(f"'y' cannot be complex")
    
    w_array = sp.signal.windows.get_window(window, ns)
    
    # rFFT
    hx = np.fft.rfft(x * w_array)
    hy = np.fft.rfft(y * w_array)
    ff = np.fft.rfftfreq(ns, d=at)

    if psd is not None:
        # Lowest and highest frequency cut-off taken from the given psd
        f_min, f_max = psd[0][[0,-1]]
        i_min = np.searchsorted(ff, f_min, side='left')
        i_max = np.searchsorted(ff, f_max, side='right')
        
        hx = hx[i_min:i_max]
        hy = hy[i_min:i_max]
        ff = ff[i_min:i_max]
    
    af = ff[1]
    
    # Compute (x|y)
    if psd is None:
        inner = 4 * af * np.sum(hx * hy.conj()).real
    else:
        psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
        inner = 4 * af * np.sum((hx * hy.conj()) / psd_interp).real
    
    return inner


def overlap(x, y, *, at=1, psd=None, window=('tukey', 0.5)):
    """Compute the Overlap between two signals:
        O = (x|y) / sqrt((x|x) · (y|y))

    Reference values:

         1 → Perfect correlation.
         0 → No correlation.
        -1 → Perfect anticorrelation.

    Parameters
    ----------
    x, y: array
        Signals to compare.

    at: float
        Sample time step. Leave as `at=1` if signals in whitened space.

    psd: 2d-array, optional
        PSD to weight the overlap, will be linearly interpolated to the right
        frequencies.

            psd[0] = frequencies
            psd[1] = psd samples
    
    References
    ----------
    [1]: Badger C. et al., 2022 (10.48550/arxiv.2210.06194)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    inner = lambda a, b: inner_product_weighted(a, b, at=at, psd=psd, window=window)

    with np.errstate(divide='ignore', invalid='ignore'):
        overlap = inner(x, y) / np.sqrt(inner(x, x) * inner(y, y))
        overlap = np.nan_to_num(overlap)

    return float(overlap)


def doverlap(x, y, *, at, psd=None, window=('tukey', 0.5)):
    """Compute the Overlap pseudo-distance.
    
    Useful to use the overlap as loss function.

    Reference values:

        0 → Perfect correlation.
        ½ → No correlation.
        1 → Perfect anticorrelation.

    Parameters
    ----------
    x, y: array
        Signals to compare.

    at: float
        Sample time step.

    psd: 2d-array, optional
        PSD to weight the overlap, will be linearly interpolated to the right
        frequencies.

            psd[0] = frequencies
            psd[1] = psd samples
    """
    return (1 - overlap(x, y, at=at, psd=psd, window=window)) / 2


def match(x, y, *, at=1, psd=None, window=('tukey', 0.5), return_lag=False):
    """Time/phase–maximised match between two (whitened) signals.

    This computes the PSD-weighted, normalised inner product maximised over
    a cyclic time shift (lag) and over phase (by taking the absolute value).

    TODO: The values don't seem to match exactly PyCBC's `match` function. Check.

    Parameters
    ----------
    x, y : ndarray
        Signals to compare (same length).
    at : float
        Sample time step (seconds).
    psd : 2d-array, optional
        If given, weights the frequency-domain inner product; linearly
        interpolated to FFT frequencies. psd[0]=freqs, psd[1]=PSD samples.
        If None, the signal is assumed to be whitened, and therefore `at` can
        be left `at=1` since it cancels out.
    window : str | tuple, optional
        Any scipy.signal window spec; applied equally to x and y.
    return_lag : bool, optional
        If True, also return (lag_samples, lag_seconds) at which the maximum
        match is attained (cyclic correlation notion).

    Returns
    -------
    m : float
        Match in [0, 1].
    (k, tau) : tuple[int, float], optional
        Index lag and time lag in seconds (only if return_lag=True).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(x)
    if n != len(y):
        raise ValueError("both 'x' and 'y' must be of the same length")
    if not np.isrealobj(x) or not np.isrealobj(y):
        raise ValueError("inputs must be real-valued")

    # Window and FFT
    w = sp.signal.windows.get_window(window, n)
    Xf = np.fft.rfft(x * w)
    Yf = np.fft.rfft(y * w)
    ff = np.fft.rfftfreq(n, d=at)

    # Optional band-limit from PSD
    if psd is not None:
        f_min, f_max = psd[0][[0, -1]]
        i_min = np.searchsorted(ff, f_min, side='left')
        i_max = np.searchsorted(ff, f_max, side='right')
    else:
        i_min, i_max = 0, Xf.size
    if i_max - i_min <= 0:
        return (0.0, (0, 0.0)) if return_lag else 0.0

    df = ff[1] - ff[0] if ff.size > 1 else 1.0 / (n * at)

    # Weighting (flat for whitened data)
    if psd is None:
        W_band = 1.0
    else:
        psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff[i_min:i_max])
        W_band = 1.0 / psd_interp

    Xb = Xf[i_min:i_max]
    Yb = Yf[i_min:i_max]

    # Norms <x|x>, <y|y> (on the band)
    nx = 4.0 * df * np.sum((Xb * Xb.conj() * W_band).real)
    ny = 4.0 * df * np.sum((Yb * Yb.conj() * W_band).real)
    denom = np.sqrt(nx * ny)
    if not np.isfinite(denom) or denom == 0.0:
        return (0.0, (0, 0.0)) if return_lag else 0.0

    # Cross-spectrum on the band
    Xcross_band = (Xb * Yb.conj()) * W_band

    # ---- Correct one-sided scaling for irfft ----
    Z = np.zeros(n // 2 + 1, dtype=np.complex128)
    Z[i_min:i_max] = (4.0 * df / denom) * Xcross_band

    # Halve all interior bins; keep DC (0) and Nyquist (if present) unhalved
    weights = np.full_like(Z, 0.5, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    Z *= weights
    # ---------------------------------------------

    # Correlation vs cyclic lag (irfft has 1/n)
    cc = np.fft.irfft(Z, n=n) * n

    k = int(np.argmax(np.abs(cc)))
    m = float(np.abs(cc[k]))
    # tiny numerical guard
    if 1.0 < m < 1.0 + 1e-10:
        m = 1.0

    if return_lag:
        # map to signed lag
        k_signed = k if k <= n // 2 else k - n
        tau = k_signed * at
        return m, (k_signed, tau)
    return m


def imatch(x, y, *, at=1, psd=None, window=('tukey', 0.5), return_lag=False):
    """Shorthand for `1 - match()`."""
    return 1 - match(x, y, at=at, psd=psd, window=window, return_lag=return_lag)


def snr(strain, *, psd, at, window=('tukey',0.5)):
    """Signal to Noise Ratio."""
    # rFFT
    strain = np.asarray(strain)
    ns = len(strain)
    if isinstance(window, tuple):
        window = sp.signal.windows.get_window(window, ns)
    else:
        window = np.asarray(window)
    hh = np.fft.rfft(strain * window)
    ff = np.fft.rfftfreq(ns, d=at)
    af = ff[1]

    # Lowest and highest frequency cut-off taken from the given psd
    f_min, f_max = psd[0][[0,-1]]
    i_min = np.argmin(ff < f_min)
    i_max = np.argmin(ff < f_max)
    if i_max == 0:
        i_max = len(hh)
    hh = hh[i_min:i_max]
    ff = ff[i_min:i_max]

    # SNR
    psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
    sum_ = np.sum(np.abs(hh)**2 / psd_interp)
    snr = np.sqrt(4 * at**2 * af * sum_)

    return snr


def find_merger(h: np.ndarray) -> int:
    """Estimate the index position of the merger in the given strain.
    
    This could be done with a better estimation model, like a gaussian in
    the case of binary mergers. However for our current project this does not
    make much difference.
    
    """
    return np.argmax(np.abs(h))
