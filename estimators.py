import numpy as np
import scipy as sp
import scipy.signal
import scipy.interpolate


def mse(x, y):
    """Mean Squared Error."""
    return np.mean((x-y)**2) / len(x)


def medse(x, y):
    """Median Squared Error."""
    return np.median((x-y)**2)


def ssim(x, y):
    """Structural similarity index."""  # TODO: Doc més
    mux = x.mean()
    muy = y.mean()
    cov_mat = np.cov(x, y, ddof=0)
    sx2 = cov_mat[0, 0]
    sy2 = cov_mat[1, 1]
    sxy = cov_mat[0, 1]
    l_ = 1
    c1 = (0.01*l_) ** 2
    c2 = (0.03*l_) ** 2

    return ((2*mux*muy+c1) * (2*sxy+c2)
            / ((mux**2+muy**2+c1) * (sx2+sy2+c2)))


def dssim(x, y):
    """Structural dissimilarity."""
    return (1 - ssim(x, y)) / 2


def issim(x, y):
    """Inverse (also invented) structural similarity."""
    return 1 - ssim(x, y)


def residual(x, y):
    """Norm of the difference between 'x' and 'y'."""
    return np.linalg.norm(x - y)


def softmax(x, axis=None):
    """Softmax probability distribution."""
    coefs = np.exp(x)
    return coefs / coefs.sum(axis=axis, keepdims=True)


def _weighted_inner(x, y, psd, at, window):
    """Compute the (x|y) between two signals as described in Ref.
    
    Note: The coefficients are ommited since they cancel themselve in the overlap computation.
    
    x, y: array
        Signals to compare.
    
    psd: 2d-array
        PSD to weight the overlap, will be linearly interpolated to the right frequencies.
        psd[0] = frequencies
        psd[1] = psd samples
    
    Ref: Eq. 12, DOI: 10.48550/arxiv.2210.06194
    
    """    
    ns = len(x)
    if ns != len(y):
        raise ValueError("both 'x' and 'y' must be of the same length")
    window = sp.signal.windows.get_window(window, ns)
    
    # rFFT
    hx = np.fft.rfft(x * window)
    hy = np.fft.rfft(y * window)
    ff = np.fft.rfftfreq(ns, d=at)

    # Lowest and highest frequency cut-off taken from the given psd
    f_min, f_max = psd[0][[0,-1]]
    i_min = np.argmax(ff >= f_min)
    i_max = np.argmax(ff <= f_max)
    if i_max == 0:
        i_max = len(ff)
    hx = hx[i_min:i_max]
    hy = hy[i_min:i_max]
    ff = ff[i_min:i_max]
    af = ff[1]
    
    # Compute (x|y)
    psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
    xy = np.sum((hx*hy.conj() + hx.conj()*hy) / psd_interp).real
    
    return xy


def overlap(x, y, psd, at, window=('tukey', 0.5)):
    """Compute the Overlap between two signals:
        Ov = (x|y) / sqrt((x|x) · (y|y))

    x, y: array
        Signals to compare.

    psd: 2d-array
        PSD to weight the overlap, will be linearly interpolated to the right frequencies.
        psd[0] = frequencies
        psd[1] = psd samples

    at: float
        Time step, inverse of sampling rate of 'x' and 'y'.
    
    Ref: Badger C. et al., 2022 (10.48550/arxiv.2210.06194)
    
    """
    x = np.asarray(x)
    y = np.asarray(y)
    wei = lambda a, b: _weighted_inner(a, b, psd, at, window)

    with np.errstate(divide='ignore', invalid='ignore'):
        overlap = wei(x, y) / np.sqrt(wei(x, x) * wei(y, y))
        np.nan_to_num(overlap, copy=False)

    return overlap


def ioverlap(x, y, psd, at, window=('tukey', 0.5)):
    """Compute `1 - Overlap()`."""

    return 1 - overlap(x, y, psd, at, window=window)


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
    i_min = np.argmax(ff >= f_min)
    i_max = np.argmax(ff <= f_max)
    # If max(psd[0]) > max(ff) do not cut off
    if i_max == 0:
        i_max = len(hh)
    hh = hh[i_min:i_max]
    ff = ff[i_min:i_max]

    # SNR
    psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
    sum_ = np.sum(np.abs(hh)**2 / psd_interp)
    snr = np.sqrt(4 * at**2 * af * sum_)

    return snr
