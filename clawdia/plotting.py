"""Ad-hoc plotting functions for visualising results.

This module includes a variety of plotting utilities designed to help visualise 
and interpret results during the development and debugging of the CLAWDIA
pipeline. 

While not essential for CLAWDIA's core processing, these functions are useful
for presenting and analyzing outcomes, such as confusion matrices, dictionary
atoms, and spectrograms.

"""
from colorsys import rgb_to_hls, hls_to_rgb
import itertools as it

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(cmat, ax=None, labels=None, mode='both', vmin=None, vmax=None,
                   cmap="PaleBlues", **kwargs):
    """Plot a confusion matrix.

    Plot a pre-computed confusion matrix `cmat`.
    Rows must contain true values, and columns predicted values. For example,
    in a binary classification case:

    +---------+---------+---------+
    |  T \ P  | Pred C1 | Pred C2 |
    +=========+=========+=========+
    | True C1 | TP      | FN      |
    +---------+---------+---------+
    | True C2 | FP      | TN      |
    +---------+---------+---------+

    Where:

    - **TP** (True Positives): Correctly predicted Class 1.
    - **FN** (False Negatives): Class 1 incorrectly predicted as Class 2.
    - **FP** (False Positives): Class 2 incorrectly predicted as Class 1.
    - **TN** (True Negatives): Correctly predicted Class 2.

    Parameters
    ----------
    cmat : array-like of shape (n_classes, n_classes)
        The confusion matrix to plot.
    
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the matrix. If not given, a new figure and axes
        are created.
    
    labels : list of str, optional
        The labels for the classes. If not given, the integers from 0 to
        `n_classes-1` are used.
    
    mode : {'absolute', 'percent', 'both'}
        The format of the annotations: absolute numbers, percentages, or both.
        Defaults to 'both'.
    
    vmin : float, optional
        The minimum value of the color scale.
    
    vmax : float, optional
        The maximum value of the color scale.

    cmap : Function (matplotlib's Colormap equivalent) | str
        Defaults to "PaleBlue", a custom modification of Matplotlib's "Blues".
    
    **kwargs
        Additional keyword arguments are passed to `matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure containing the plot, or None if `ax` was given.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        cmat_perc = np.nan_to_num(cmat / np.sum(cmat, axis=1, keepdims=True))
    
    if mode == 'both':
        if plt.rcParams["text.usetex"]:
            format_str = lambda *args: '{}\n{:.0%}'.format(*args).replace('%', r'\,\%')
        else:
            format_str = lambda *args: '{}\n{:.0%}'.format(*args)
    elif mode == 'percent':
        if plt.rcParams["text.usetex"]:
            format_str = lambda *args: '{:.0%}'.format(args[1]).replace('%', r'\,\%')
        else:
            format_str = lambda *args: '{:.0%}'.format(args[1])
    elif mode == 'absolute':
        format_str = lambda *args: str(args[0])
    else:
        raise ValueError("mode can only be 'absolute', 'percent' or 'both'")
    
    if not callable(cmap):
        if cmap == "PaleBlues":
            cmap = _desaturate_cmap(plt.cm.Blues, desaturation_factor=0.8, brightness_boost=0.2)
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        else:
            raise TypeError("Colormap not recognized")

    n_classes = len(cmat)

    cmat_str = np.empty_like(cmat_perc, dtype=object)
    for i_true, i_pred in it.product(range(n_classes), repeat=2):
        cmat_str[i_true, i_pred] = format_str(cmat[i_true, i_pred], cmat_perc[i_true, i_pred])

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    ax.imshow(cmat_perc, cmap=cmap, vmin=vmin, vmax=vmax)
    
    for i_true, i_pred in it.product(range(n_classes), repeat=2):
        ax.annotate(
            cmat_str[i_true, i_pred],
            xy=(i_pred, i_true),
            ha='center',
            va='center'
        )
    
    ax.grid(False)

    # Change the frame (axes spines) color to gray
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    # Set X and Y labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Set ticks
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    
    ax.set_xlim([-0.5, n_classes-0.5])
    ax.set_ylim([n_classes-0.5, -0.5])  # Invert Y-axis

    if labels is not None:
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=45)

    # Move X-axis ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    try:
        return fig
    except NameError:
        return None


def plot_dictionary(array, c=None, ylim=None, **plot_kw):
    """Plot atoms from a dictionary in a squared matrix.
    
    Parameters
    ----------
    array : 2d-array
        Dictionary matrix with shape (a, l).
    
    c : int, optional
        Number of atoms at each side of the squared matrix of plots; the total
        number of plotted atoms will be `c ** 2`.
        If not given, it is computed as `int(np.sqrt(a))`.
    
    **plot_kw : optional
        Passed to pyplot.subplots().
    
    """
    if c is None:
        c = int(np.sqrt(array.shape[0]))
    fig, axs = plt.subplots(ncols=c, nrows=c, **plot_kw)
    for i in range(c**2):
        ax = axs[i//c,i%c]
        ax.plot(array[i], lw=1)
        ax.set_ylim(ylim)
        ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    
    return fig


def _desaturate_cmap(cmap, *, desaturation_factor, brightness_boost):
    """
    Desaturate and brighten a colormap.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to modify.
    desaturation_factor : float
        The factor by which to reduce the saturation (0 = grayscale, 1 = no change).
    brightness_boost : float
        The amount to increase the brightness (lightness) (0 = no change).

    Returns
    -------
    LinearSegmentedColormap
        The modified colormap with reduced saturation and increased brightness.
    """
    colors = cmap(np.linspace(0, 1, 256))  # Extract RGB colors
    modified_colors = []

    for r, g, b, a in colors:  # Process each RGBA color
        h, l, s = rgb_to_hls(r, g, b)  # Convert to HLS
        s *= desaturation_factor  # Reduce saturation
        l = min(l + brightness_boost * (1 - l), 1.0)  # Adjust brightness, ensure no overflow
        r, g, b = hls_to_rgb(h, l, s)  # Convert back to RGB
        modified_colors.append((r, g, b, a))  # Add the alpha channel back

    return mpl.colors.LinearSegmentedColormap.from_list("DesaturatedBrightenedBlues", modified_colors)