import itertools as it

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def plot_confusion(cmat, ax=None, labels=None, mode='both', vmin=None, vmax=None):
    if mode in ('absolute', 'both'):
        with np.errstate(divide='ignore', invalid='ignore'):
            cmat_perc = np.nan_to_num(cmat / np.sum(cmat, axis=1, keepdims=True))
        if mode == 'absolute':
            format_str = lambda *args: str(args[0])
        else:
            format_str = lambda *args: '{}\n{:.0%}'.format(*args)
    elif mode == 'percent':
        cmat_perc = cmat
        format_str = lambda *args: '{:.0%}'.format(args[0])
    else:
        raise ValueError("mode can only be 'absolute', 'percent' or 'both'")

    vmin = 0 if vmin is None else vmin
    vmax = 1.2 if vmax is None else vmax
    n_classes = len(cmat)
    cmat_str = np.empty_like(cmat_perc, dtype=object)
    for i, j in it.product(range(n_classes), repeat=2):
        cmat_str[i,j] = format_str(cmat[i,j], cmat_perc[i,j])

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.imshow(cmat_perc, cmap=plt.get_cmap('Blues'), vmin=vmin, vmax=vmax)
    for i, j in it.product(range(n_classes), repeat=2):
        ax.annotate(cmat_str[i,j], xy=(j,i), ha='center', va='center')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_xlim([-0.5, n_classes-0.5])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(labels if labels is not None else None)
    ax.set_ylabel("Actual")
    ax.set_ylim([-0.5, n_classes-0.5])
    ax.invert_yaxis()
    ax.grid(False)

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
