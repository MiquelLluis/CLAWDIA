import numpy as np


def mse(x, y):
    """Mean Squared Error."""
    return np.mean((x-y)**2)


def ssim(x, y):
    """Structural similarity index."""  # TODO: Doc més
    mux = x.mean()
    muy = y.mean()
    sx2 = x.var()
    sy2 = y.var()
    sxy = np.cov(x, y, ddof=0)[0, 1]
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
