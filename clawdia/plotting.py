import itertools as it

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def plot_confusion(cmat, ax=None, labels=None, mode='both', invert_axis=False, vmin=None, vmax=None, **kwargs):
    """Plot a confusion matrix.

    Parameters
    ----------
    cmat : array-like of shape (n_classes, n_classes)
        The confusion matrix to plot, with true label being i-th class and
        predicted label being j-th class.
    
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the matrix. If not given, a new figure and axes
        are created.
    
    labels : list of str, optional
        The labels for the classes. If not given, the integers from 0 to
        `n_classes-1` are used.
    
    mode : {'absolute', 'percent', 'both'}
        The format of the annotations: absolute numbers, percentages, or both.
        Defaults to 'both'.
    
    invert_axis : bool
        If True, invert the axes of the plot (predicted values at the abscissa).
        Defaults to False (predicted values at the ordinate).
    
    vmin : float, optional
        The minimum value of the color scale.
    
    vmax : float, optional
        The maximum value of the color scale.
    
    **kwargs
        Additional keyword arguments are passed to `matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure containing the plot, or None if `ax` was given.

    """
    if invert_axis:
        cmat = cmat.T
        axis = 1
    else:
        axis = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        cmat_perc = np.nan_to_num(cmat / np.sum(cmat, axis=axis, keepdims=True))
    
    if mode == 'both':
        format_str = lambda *args: '{}\n{:.0%}'.format(*args)
    elif mode == 'percent':
        format_str = lambda *args: '\n{:.0%}'.format(args[1])
    elif mode == 'absolute':
        format_str = lambda *args: str(args[0])
    else:
        raise ValueError("mode can only be 'absolute', 'percent' or 'both'")

    n_classes = len(cmat)

    cmat_str = np.empty_like(cmat_perc, dtype=object)
    for i_true, i_pred in it.product(range(n_classes), repeat=2):
        cmat_str[i_true,i_pred] = format_str(cmat[i_true,i_pred], cmat_perc[i_true,i_pred])

    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    
    if vmax is None:
        vmax = 1.2
    ax.imshow(cmat_perc, cmap=plt.get_cmap('Blues'), vmin=vmin, vmax=vmax)
    
    for i_true, i_pred in it.product(range(n_classes), repeat=2):
        ax.annotate(cmat_str[i_true,i_pred], xy=(i_pred,i_true), ha='center', va='center')
    
    ax.grid(False)

    if invert_axis:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    else:
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    
    ax.set_xlim([-0.5, n_classes-0.5])
    ax.set_ylim([-0.5, n_classes-0.5])

    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    try:
        return fig
    except NameError:
        return None


def plot_dictionary(array, c=None, **plot_kw):
    """Plot atoms from a dictionary in a squared matrix.
    
    Parameters
    ----------
    array : 2d-array
        Dictionary matrix in Fortran order with shape (l, a).
    
    c : int, optional
        Number of atoms at each side of the squared matrix of plots; the total
        number of plotted atoms will be `c ** 2`.
        If not given, it is computed as `int(np.sqrt(a))`.
    
    **plot_kw : optional
        Passed to pyplot.subplots().
    
    """
    if c is None:
        c = int(np.sqrt(array.shape[1]))
    fig, axs = plt.subplots(ncols=c, nrows=c, **plot_kw)
    for i in range(c**2):
        ax = axs[i//c,i%c]
        ax.plot(array[:,i], lw=1)
        ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    
    return fig


def plot_spec_of(strain, figsize=(10,5), sf=4096, window='hann', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    ff, tt, spec = sp.signal.spectrogram(
        strain,
        fs=sf,
        window=window,
        nperseg=256,
        nfft=2*sf,
        noverlap=256-32
    )
    norm = mpl.colors.LogNorm(clip=False, vmin=vmin, vmax=vmax)
    pcm = ax.pcolormesh(tt, ff, spec, norm=norm)

    return fig, spec, pcm
