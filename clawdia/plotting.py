"""Ad-hoc plotting functions for visualizing results.

This module includes a variety of plotting utilities designed to help visualize 
and interpret results during the development and debugging of the CLAWDIA
pipeline. 

While not essential for CLAWDIA's core processing, these functions are useful
for presenting and analyzing outcomes, such as confusion matrices, dictionary
atoms, and spectrograms.

"""
from colorsys import rgb_to_hls, hls_to_rgb
import itertools as it
import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

# from gwadama.fat import instant_frequency  # IMPORTED INSIDE THE SPECTROGRAM FUNCTION


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
        ax.annotate(cmat_str[i_true, i_pred], xy=(i_pred, i_true), ha='center', va='center')
    
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
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

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
        Dictionary matrix in Fortran order with shape (a, l).
    
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


def plot_spectrogram_with_instantaneous_features(
        strain_array, time_array, sampling_rate=2**14, outseg=None, outfreq=None,
        window=sp.signal.windows.tukey(128,0.5), hop=32, mfft=2**14, vmin=-22, ax=None):
    """Plot the spectrogram, instantaneous frequency, and strain's waveform.

    This function generates a multi-panel plot consisting of:
    
    1. A spectrogram of the gravitational wave strain, obtained using
       Short-Time Fourier Transform (STFT), visualizing the frequency evolution
       over time.
    2. The instantaneous frequency of the strain, plotted on top of the
       spectrogram to show the frequency changes in real time.
    3. The raw gravitational wave strain in the time domain, shown above the
       spectrogram for direct comparison.

    Key features of the plot:

    - **Spectrogram**: The frequency content of the gravitational wave signal
      is displayed over time using a color map (`inferno`), with the x-axis
      representing time (in milliseconds) and the y-axis representing
      frequency (in Hz).
    - **Instantaneous Frequency**: Plots the instantaneous frequency of the
      strain over time, highlighting the frequency variations.
    - **Energy Normalization**: The spectrogram uses a logarithmic scale for
      the energy (power spectral density, PSD), normalized by the maximum
      energy value in the signal.
    - **Dynamic Range Control**: The color scale of the spectrogram can be
      adjusted via the `vmin` parameter to emphasize specific energy levels.
    - **Time-Domain Waveform**: A plot of the original strain data in the time
      domain is shown above the spectrogram, providing context for the signal's
      evolution.
    - **Segmentation**: The user can specify the time (`outseg`) and frequency
      (`outfreq`) ranges to focus on specific parts of the data.
    - **Customization**: The plot has a black background, white grid lines,
      and labeled colorbars for clarity.

    Parameters
    ----------
    strain_array : numpy.ndarray
        The time-domain strain data of the gravitational wave signal.
    
    time_array : numpy.ndarray
        Array of time stamps corresponding to the strain data.
    
    sampling_rate : int, optional
        The sampling rate of the data in Hz (default is 2^14, or 16384 Hz).
    
    outseg : tuple, optional
        A tuple specifying the time range (start, end) in seconds for the
        x-axis. If `None`, the entire time range of the input data is used.
    
    outfreq : tuple, optional
        A tuple specifying the frequency range (start, end) in Hz for the
        y-axis. If `None`, the full frequency range (up to Nyquist frequency)
        is used.
    
    window : numpy.ndarray, optional
        The window function applied during STFT computation (default is a
        Tukey window).
    
    hop : int, optional
        The hop size between successive STFT windows (default is 32).
    
    mfft : int, optional
        The number of points in the FFT used for STFT computation (default
        is 2^14).
    
    vmin : float, optional
        The minimum value for the color scale in the spectrogram (default
        is -22). This controls the dynamic range of the color map.
    
    ax : matplotlib.axes.Axes, optional
        The axes object on which to plot the spectrogram. If `None`, a new
        figure and axes are created.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the complete plot.
    
    axs : List[matplotlib.axes.Axes]
        A list of axes objects containing the spectrogram, the colorbar, and
        the time-domain plots.
    
    Sxx : numpy.ndarray
        The computed spectrogram (PSD values) of the input strain data.

    Notes
    -----
    - The spectrogram normalization is performed on the square root of the 
      Power Spectral Density (PSD), converted to a logarithmic scale.
    - The y-axis of the spectrogram uses a kilohertz scale for readability.
    - The time-domain waveform is plotted without axes labels for simplicity.
    - Instantaneous frequency values below zero are masked to avoid displaying 
      non-physical results.
    
    """
    from gwadama.fat import instant_frequency

    # Compute the spectrogram using the ShortTimeFFT class.
    stfft_model = sp.signal.ShortTimeFFT(
        win=window, hop=hop, fs=sampling_rate, mfft=mfft,
        fft_mode='onesided', scale_to='psd'
    )
    Sxx = stfft_model.spectrogram(strain_array)
    with np.errstate(divide='ignore'):
        normalized_Sxx = np.log10(np.sqrt(Sxx))
    normalized_Sxx -= np.max(normalized_Sxx)
    t0, t1, f0, f1 = stfft_model.extent(len(strain_array))
    t_origin = time_array[0]
    t0 += t_origin
    t1 += t_origin

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # SPECTROGRAM (ax1)
    ax.imshow(normalized_Sxx, cmap='inferno', origin='lower', aspect='auto',
              extent=(t0,t1,f0,f1), interpolation='lanczos', vmin=vmin)
    # ...and Instant Frequency
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        instant_freq = instant_frequency(strain_array, sample_rate=sampling_rate)
    length = len(strain_array)
    t1_instant = t_origin + (length-1)/sampling_rate
    instant_time = np.linspace(t_origin, t1_instant, length)
    mask = instant_freq >= 0  # Remove non-physical frequencies
    instant_freq = instant_freq[mask]
    instant_time = instant_time[mask]
    ax.plot(instant_time, instant_freq, 'purple', lw=2)
    
    # COLORBAR (ax2)
    ax_pos = ax.get_position()
    ax2_width = 0.015  # Set the width of the colorbar axis
    ax2_pad = 0.01    # Set the padding between the spectrogram and colorbar
    ax2_x = ax_pos.x1 + ax2_pad
    ax2 = fig.add_axes([ax2_x, ax_pos.y0, ax2_width, ax_pos.height])
    cbar = fig.colorbar(ax.images[-1], cax=ax2)
    
    # LABELS, LIMITS, ETC
    ax.grid(True, ls='--', alpha=.4)
    # ...limits
    if outseg is None:
        ax.set_xlim(time_array[0], time_array[-1])
    else:
        ax.set_xlim(*outseg)
    if outfreq is None:
        ax.set_ylim(0, sampling_rate/2)
    else:
        ax.set_ylim(*outfreq)
    # ...labels.
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [kHz]')
    cbar.set_label('Normalized energy')
    # ...Y ticks to kHz
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))
    # ...X ticks to milliseconds and avoid roundoff errors.
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Set style first, for new ticklabels.
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    # ax.set_xticklabels(np.round(xticks * 1000).astype(int))
    ax.set_xticklabels(np.round(xticks*1000, decimals=1))
    # ...background in black to match the colormap.
    ax.set_facecolor('black')

    # GW IN TIME-DOMAIN ON TOP OF THE SPECTROGRAM (ax3)
    ax3 = fig.add_axes([ax_pos.x0, ax_pos.y0+ax_pos.height+0.03, ax_pos.width, ax_pos.height*0.2])
    ax3.plot(time_array, strain_array, c='black', lw=1, alpha=1)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_ylim(np.min(strain_array), np.max(strain_array))
    ax3.axis('off')

    # # DEBUG AXES AREA [
    # from matplotlib import patches
    # ax3_pos = ax3.get_position()
    # ax2_pos = ax2.get_position()
    # # Create a rectangle for the main axis (ax)
    # ax_rect = patches.Rectangle((ax_pos.x0, ax_pos.y0), ax_pos.width, ax_pos.height,
    #                             linewidth=3, edgecolor='r', facecolor='none', label='ax')
    # # Create a rectangle for the colorbar axis (ax2)
    # ax2_rect = patches.Rectangle((ax2_pos.x0, ax2_pos.y0), ax2_pos.width, ax2_pos.height,
    #                             linewidth=3, edgecolor='b', facecolor='none', label='ax2')
    # # Create a rectangle for the time-domain axis (ax3)
    # ax3_rect = patches.Rectangle((ax3_pos.x0, ax3_pos.y0), ax3_pos.width, ax3_pos.height,
    #                             linewidth=3, edgecolor='g', facecolor='none', label='ax3')
    # # Add the rectangles to the figure
    # fig.add_artist(ax_rect)
    # fig.add_artist(ax2_rect)
    # fig.add_artist(ax3_rect)
    # # ]
    
    return fig, [ax, ax2, ax3], Sxx


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