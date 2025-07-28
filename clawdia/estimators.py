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
        i_max = np.searchsorted(ff, f_max, side='left')
        
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


def overlap(x, y, *, at, psd=None, window=('tukey', 0.5)):
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
        Sample time step.

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
        np.nan_to_num(overlap, copy=False)

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
