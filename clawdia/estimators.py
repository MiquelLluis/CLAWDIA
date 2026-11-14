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
    return float(np.mean((x-y)**2))


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

    Notes
    -----
    The stabilising constants currently assume a signal data range of one.
    
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

    Useful as a loss function to perform minimisation.
        
    """
    return -ssim(x, y)


def residual(x, y):
    """Norm of the difference between 'x' and 'y'."""
    return np.linalg.norm(x - y)


def softmax(x, axis=None):
    """Softmax probability distribution."""
    return sp.special.softmax(x, axis=axis)


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

    Notes
    -----
    Currently uses unscaled NumPy FFT coefficients. With the same PSD, window,
    and included frequency bins, ``(x|x)`` equals ``snr(x, ...)**2 / at**2``.
    The physical normalisation and DC/Nyquist weighting remain to be reviewed.
    
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
    af = ff[1]

    if psd is not None:
        # Lowest and highest frequency cut-off taken from the given psd
        f_min, f_max = psd[0][[0,-1]]
        i_min = np.searchsorted(ff, f_min, side='left')
        i_max = np.searchsorted(ff, f_max, side='right')
        
        hx = hx[i_min:i_max]
        hy = hy[i_min:i_max]
        ff = ff[i_min:i_max]
    
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


def match(x, y, *, at=1, psd=None, window=('tukey', 0.5), return_lag=False) -> float | tuple[float, tuple[float, float]]:
    """Cyclic time/phase-maximised match between two real-valued signals.

    Computes the PSD-weighted, normalised complex positive-frequency
    correlation, maximised over integer-sample cyclic time shifts. Its
    magnitude analytically maximises over an arbitrary global phase rotation
    of the positive-frequency components (with conjugate rotation of negative
    frequencies to keep the time-domain signals real).

    The comparison uses the windowed signals over the selected frequency band.
    0 Hz and Nyquist (for even lengths) are excluded from both the
    correlation and the norms.

    The continuous definition whose discrete approximation is computed is:

        M(x,y) = max_τϕ 〈x|y(τ,ϕ)〉 / D,       D = √(〈x|x〉 * 〈y|y〉)

    Here y(τ,ϕ) is y delayed by τ and phase-rotated by ϕ. With X(f) and Y(f)
    denoting the Fourier transforms of the windowed signals, we define the
    PSD-weighted cross-spectrum and its complex positive-frequency correlation:

        P_xy(f) = X(f) Y*(f) / S_n(f)
        C_xy(τ) = 4 ∫_[f_min,f_max] P_xy(f) exp(+2πifτ) df

    At each fixed lag τ, phase can be maximised analytically:

        max_ϕ 〈x | y(τ,ϕ)〉 = |C_xy(τ)|    ⟶    M(x,y) = max_τ |C_xy(τ)| / D

    In code, ``cross_spectrum`` contains the normalised discrete spectral
    coefficients after cancelling common factors. The resulting ``cc[j]``
    approximates ``C_xy(j * at) / D``; the time search uses cyclic sample lags.

    Parameters
    ----------
    x, y : array_like
        One-dimensional, real-valued signals of the same length (at least 3).
    at : float, optional
        Sample time step in seconds; must be positive. It only sets the
        frequency/lag scales when `psd` is None.
    psd : array_like, shape (2, M), optional
        If given, weights the frequency-domain inner product. The first row
        contains strictly increasing frequencies in Hz, the second row the
        one-sided PSD samples. The PSD is linearly interpolated onto the FFT
        frequencies, and only the PSD's frequency range is used. PSD values
        must be positive and finite at the included FFT frequencies.
        If None, flat frequency weighting is used (appropriate for signals
        consistently whitened using the same PSD).
    window : str | tuple | None, optional
        A scipy.signal window specification, applied equally to x and y
        before the FFT. If None, do not window. Windowing changes the signals
        being compared and can affect shift invariance.
    return_lag : bool, optional
        If True, also return (lag_samples, lag_seconds) at the maximum.
        Positive lag means delaying y: y_lag[j] = y[(j - lag) % n].

    Returns
    -------
    m : float
        Discrete cyclic match in [0, 1] (up to floating-point roundoff).
        Returns 0 if the analysed band is empty or either signal has zero
        norm in that band; the lag is then set to 0 by convention.
    (k, tau) : tuple[int, float], optional
        Signed cyclic lag and its value in seconds (only if return_lag=True).

    Notes
    -----
    The time maximisation is restricted to integer-sample cyclic shifts.
    No sub-sample interpolation or linear (zero-padded) correlation is used.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("'x' and 'y' must be one-dimensional")
    n = x.size
    if n != y.size:
        raise ValueError("both 'x' and 'y' must be of the same length")
    if n < 3:
        raise ValueError("signals must have at least 3 samples")
    if not np.isrealobj(x) or not np.isrealobj(y):
        raise ValueError("inputs must be real-valued")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("inputs must contain only finite values")
    try:
        at = float(at)
    except (TypeError, ValueError) as exc:
        raise ValueError("'at' must be a positive finite number") from exc
    if not np.isfinite(at) or at <= 0:
        raise ValueError("'at' must be a positive finite number")

    # Apply the same window before computing both Fourier transforms.
    if window is not None:
        w = sp.signal.windows.get_window(window, n)
        x = x * w
        y = y * w
    Xf = np.fft.rfft(x)
    Yf = np.fft.rfft(y)
    ff = np.fft.rfftfreq(n, d=at)

    # Only positive frequencies.
    # For even n, the final rfft bin is the (real-valued) Nyquist bin.
    # Exclude 0 Hz and Nyquist from the correlation and norms.
    i_min, i_max = 1, (n + 1) // 2  # upper bound is exclusive

    if psd is not None:
        psd = np.asarray(psd, dtype=np.float64)
        if psd.ndim != 2 or psd.shape[0] != 2 or psd.shape[1] < 2:
            raise ValueError("'psd' must have shape (2, M), with M >= 2")
        psd_freqs, psd_values = psd
        if (not np.all(np.isfinite(psd))
                or psd_freqs[0] < 0
                or np.any(np.diff(psd_freqs) <= 0)
                or np.any(psd_values < 0)):
            raise ValueError("PSD frequencies must increase, and PSD values "
                             "must be finite and non-negative")
        # Restrict frequency bounds to PSD's available range:
        i_min = max(i_min, int(np.searchsorted(ff, psd_freqs[0], side='left')))
        i_max = min(i_max, int(np.searchsorted(ff, psd_freqs[-1], side='right')))

    if i_max <= i_min:
        return (0.0, (0, 0.0)) if return_lag else 0.0

    # Flat weighting for whitened signals; otherwise interpolate the PSD.
    if psd is None:
        W_band = 1.0
    else:
        psd_interp = np.interp(ff[i_min:i_max], psd_freqs, psd_values)
        if np.any(~np.isfinite(psd_interp)) or np.any(psd_interp <= 0):
            raise ValueError("PSD must be finite and strictly positive "
                             "at the included FFT frequencies")
        W_band = 1.0 / psd_interp

    # X's and Y's spectral points to be included.
    Xb = Xf[i_min:i_max]
    Yb = Yf[i_min:i_max]

    # Compute the denominator: √(〈x|x〉 * 〈y|y〉), up to common factors.
    # The common 4*df factor cancels from correlation and both norms.
    nx = np.sum(np.abs(Xb)**2 * W_band)  # ~ 〈x|x〉
    ny = np.sum(np.abs(Yb)**2 * W_band)  # ~ 〈y|y〉
    if nx == 0.0 or ny == 0.0:
        return (0.0, (0, 0.0)) if return_lag else 0.0
    denominator = np.sqrt(nx) * np.sqrt(ny)
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("non-finite PSD-weighted signal norm")

    # Normalised positive-frequency cross-spectrum.
    cross_spectrum = np.zeros(n, dtype=np.complex128)
    cross_spectrum[i_min:i_max] = (Xb * Yb.conj()) * W_band / denominator

    # Undo the IFFT's 1/n factor to obtain cc[j], approximating C_xy(j * at) / D.
    cc = np.fft.ifft(cross_spectrum) * n
    k = int(np.argmax(np.abs(cc)))
    m = float(np.abs(cc[k]))
    if 1.0 < m < 1.0 + 1e-10:  # guard for numerical roundoff only
        m = 1.0

    if return_lag:
        k_signed = k if k <= n // 2 else k - n
        return m, (k_signed, k_signed * at)
    return m


def imatch(x, y, *, at=1, psd=None, window=('tukey', 0.5), return_lag=False
        ) -> float | tuple[float, tuple[float, float]]:
    """Shorthand for `1 - match()`."""
    result = match(
        x, y, at=at, psd=psd, window=window, return_lag=return_lag
    )
    if return_lag:
        value, lag = result
        return 1 - value, lag
    return 1 - result


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
