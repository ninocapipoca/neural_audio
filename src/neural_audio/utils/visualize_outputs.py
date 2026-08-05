import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from scipy.io import wavfile

def plot_spectrogram(matrix: np.ndarray,
                      time_points: np.ndarray,
                      frequencies: np.ndarray,
                      title: str='Spectrogram') -> None:
    """Plots a spectrogram-like matrix (e.g. an audiogram such as the output of `wav2aud`) on a log-scaled
    frequency axis, with magnitude converted to decibels.

    The frequency axis is displayed on a base-2 log scale so that octave spacing appears linear,
    which is useful for comparing against auditory/cortical representations where frequency channels
    are log-spaced. ``pcolormesh`` is used rather than ``imshow`` so that linearly-spaced FFT bins
    can be mapped onto this log-scaled axis correctly.

    This function draws onto the current matplotlib axes and does not create a new figure or call
    ``plt.show()`` itself, so it can be used with ``plt.subplot`` to place multiple spectrograms side
    by side or stacked, e.g.:

    .. code-block:: python

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plot_spectrogram(matrix=spectrogram, time_points=time_points_spectrogram,
                          frequencies=frequencies_spectrogram,
                          title="Regular Spectrogram")

        plt.subplot(2, 1, 2)
        plot_spectrogram(matrix=audiogram, time_points=time_points_audiogram,
                          frequencies=frequencies_audiogram,
                          title="Audiogram")

        plt.tight_layout()
        plt.show()

    :param matrix: 2-D array of magnitude values to plot, of shape (len(time_points), len(frequencies)),
        matching the (time points, frequencies) orientation of `wav2aud`'s audiogram output.
        Values are converted to decibels internally; values less than or equal to zero are clipped to
        a small positive constant beforehand to avoid taking the log of zero.
    :type matrix: numpy.ndarray

    :param time_points: 1-D array of time values (in seconds) corresponding to the rows of ``matrix``.
    :type time_points: numpy.ndarray

    :param frequencies: 1-D array of frequency values (in Hz) corresponding to the columns of ``matrix``.
    :type frequencies: numpy.ndarray

    :param title: Title displayed above the plot.
    :type title: str, optional, default='Spectrogram'

    :returns: None. The spectrogram is drawn on the current matplotlib axes.
    :rtype: None
    """

    ylim=[frequencies[0], frequencies[-1]]

    def to_decibel(x: np.ndarray) -> np.ndarray:
        return 20 * np.log10(np.maximum(x, 1e-9))

    plt.title(title)

    # pcolormesh so the linearly-spaced FFT bins can be mapped onto a log-scaled frequency axis
    plt.pcolormesh(time_points, frequencies, to_decibel(matrix).T) # NOTE - added transpose here
    plt.colorbar(label='Magnitude (dB)')

    plt.xlabel("Time (s)")
    plt.yscale('log', base=2)
    plt.ylim(*ylim)
    plt.ylabel("Frequency (Hz)")
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())


def cr_projections(cr, rates):
    """Compute the three 2-D projections of a cortical representation.

    The magnitude ``|cr|`` is averaged over the two sweep directions and over the time
    axis, then collapsed onto each remaining pair of axes. This is the shared computation
    behind :func:`plot_cr_projection`; it is exposed separately so a cortical
    representation can be reduced to its (small) projections without also plotting.

    :param cr: 4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
    :type cr: numpy.ndarray
    :param rates: Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    :type rates: np.ndarray

    :returns: Tuple ``(scale_rate, scale_freq, rate_freq)`` of 2-D arrays.
    :rtype: tuple
    """
    n_rate = len(rates)
    # magnitude, sweep directions averaged, then time-averaged -> [scale, rate, freq]
    cr_avgr = ((np.abs(cr[:, :n_rate]) + np.abs(cr[:, n_rate:2 * n_rate])) / 2).mean(axis=2)
    return cr_avgr.mean(2), cr_avgr.mean(1), cr_avgr.mean(0)


def plot_cr_projection(cr, rates, scales=None, frequencies=None, figsize=(12, 4), axes=None):
    """
    Plots 2D projections (scale-rate, scale-frequency, rate-frequency) of a cortical representation produced by `aud2cor`, with time averaged out.

    :param cr: 4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
        num_time and num_freq may include zero/wrap-around margins added by aud2cor's
        tp_margin / sp_margin (note that margin columns/rows are shown
        but left unlabeled since they have no physical meaning).
    :type cr: numpy.ndarray
    :param rates: Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    :type rates: np.ndarray
    :param scales: Scale vector used in aud2cor (for real tick labels). If None, the axis
        shows the index.
    :type scales: np.ndarray, optional
    :param frequencies: Characteristic frequencies from wav2aud (for real tick labels). If
        None, the axis shows the index. Length is expected to be <= cr.shape[3]; if cr's
        frequency axis is wider (because sp_margin > 0 was used in aud2cor), the extra
        margin columns are included in the plot but left unlabeled.
    :type frequencies: np.ndarray, optional
    :param figsize: Figure size, used only when a new figure is created (``axes=None``).
    :type figsize: tuple, optional, default=(12, 4)
    :param axes: Optional sequence of exactly 3 existing axes to draw the
        (scale-rate, scale-frequency, rate-frequency) panels into -- e.g. one row of a
        larger subplot grid, so several representations can be compared in a single figure.
        If None, a new 1x3 figure is created.
    :type axes: sequence of matplotlib.axes.Axes, optional

    :returns: The figure and its three axes
        ``(fig, (ax_scale_rate, ax_scale_freq, ax_rate_freq))``.
    :rtype: tuple
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")
    if cr.shape[1] != 2 * len(rates):
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal 2*len(rates) ({2*len(rates)})."
        )

    scale_rate, scale_freq, rate_freq = cr_projections(cr, rates)

    # Frequency margin offset: real frequency i sits at data-column i + dM
    n_freq_actual = scale_freq.shape[1]
    if frequencies is not None:
        dM = (n_freq_actual - len(frequencies)) // 2
        if dM < 0:
            raise ValueError(
                f"len(frequencies) ({len(frequencies)}) exceeds cr's frequency "
                f"dimension ({n_freq_actual}); frequencies must correspond to the "
                f"unpadded axis used in aud2cor."
            )
    else:
        dM = 0

    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        axes = np.atleast_1d(axes).ravel()
        if axes.size != 3:
            raise ValueError("axes must contain exactly 3 Axes "
                             "(scale-rate, scale-frequency, rate-frequency).")
        fig = axes[0].figure

    ax1, ax2, ax3 = axes
    panels = [
        (ax1, scale_rate, "Rate [Hz]", "Scale [cyc/oct]", "Scale-Rate"),
        (ax2, scale_freq, "Frequency [Hz]", "Scale [cyc/oct]", "Scale-Frequency"),
        (ax3, rate_freq, "Frequency [Hz]", "Rate [Hz]", "Rate-Frequency"),
    ]
    for ax, data, xlabel, ylabel, title in panels:
        im = ax.imshow(data, aspect='auto', origin='lower', cmap="viridis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    # Frequency ticks (x-axis on ax2, ax3), offset into the real-data region
    if frequencies is not None:
        f_idx = np.linspace(0, len(frequencies) - 1, 6).astype(int)
        for ax in (ax2, ax3):
            ax.set_xticks(f_idx + dM)
            ax.set_xticklabels([f"{frequencies[i]:.0f}" for i in f_idx], rotation=45)

    # Rate ticks: x-axis on ax1, y-axis on ax3 (rates aren't padded)
    r_idx = np.linspace(0, len(rates) - 1, 6).astype(int)
    r_labels = [f"{rates[i]:.1f}" for i in r_idx]
    ax1.set_xticks(r_idx); ax1.set_xticklabels(r_labels)
    ax3.set_yticks(r_idx); ax3.set_yticklabels(r_labels)

    # Scale ticks: y-axis on ax1, ax2 (scales aren't padded)
    if scales is not None:
        s_idx = np.linspace(0, len(scales) - 1, 6).astype(int)
        s_labels = [f"{scales[i]:.1f}" for i in s_idx]
        for ax in (ax1, ax2):
            ax.set_yticks(s_idx); ax.set_yticklabels(s_labels)

    if created:
        fig.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_cr_temporal(cr, rates, scales=None, frequencies=None, time_points=None,
                     figsize=(15, 4), axes=None):
    """Plots time-resolved projections (rate-time, scale-time, frequency-time) of a cortical representation produced by `aud2cor`.

    Rather than collapsing the time axis as in :func:`plot_cr_projection`, this function instead keeps time on the x-axis
    and collapses the two axes that are not being examined, producing three time-resolved views:

    - Rate-Time: how temporal-modulation (rate) energy changes over time
      (averaged over scale and frequency).
    - Scale-Time: how spectral-modulation (scale) energy changes over time
      (averaged over rate and frequency).
    - Frequency-Time: how per-channel energy changes over time
      (averaged over scale and rate).

    The magnitude ``|cr|`` is used, since it acts as a strength-of-match measure for a pair
    (scale, rate), essentially quantifying how much the signal is acting like (scale, rate).

    By default, ``cr`` is expected to contain both sweep directions (``cr.shape[1] == 2*len(rates)``),
    and the two directions are averaged together, matching
    :func:`plot_cr_projection` for comparability. To inspect a single direction instead, an appropriate
    slice of the cortical representation can be passed in, e.g:

    .. code-block:: python
    
            directionA = cr[:, :len(rates)]
            directionB = cr[:, len(rates):]
            plot_cr_temporal(directionA, rates, scales, frequencies, time_points)

    Here 'upwards' or 'downwards' are not explicitly assigned as it depends on the sign convention of 
    the rate. The naming is arbitrary; the importance lies in the filters being able to capture both
    modulation directions.


    :param cr: 4-D cortical output. Either both directions, shape
        ``(num_scales, num_rates*2, num_time, num_freq)`` (averaged together), or a single
        direction, shape ``(num_scales, num_rates, num_time, num_freq)`` (used as-is).
        ``num_time`` / ``num_freq`` may include the margins added by ``aud2cor``'s
        ``tp_margin`` / ``sp_margin``; margin rows/columns are shown but left unlabeled.
    :type cr: numpy.ndarray
    :param rates: Rate vector used in ``aud2cor`` (length ``= num_rates``).
    :type rates: np.ndarray
    :param scales: Scale vector used in ``aud2cor`` (for real y-tick labels on the
        Scale-Time panel). If ``None``, the axis shows the channel index.
    :type scales: np.ndarray, optional
    :param frequencies: Characteristic frequencies from ``wav2aud`` (for real y-tick labels
        on the Frequency-Time panel). If ``None``, the axis shows the channel index.
    :type frequencies: np.ndarray, optional
    :param time_points: Time values (in seconds) for the *unpadded* time frames, e.g. the
        ``time_points`` returned by ``wav2aud``. Used for real x-tick labels; when shorter
        than ``cr``'s time axis (because ``tp_margin > 0`` was used) the labels are offset
        into the real-data region. If ``None``, the axis shows the frame index.
    :type time_points: np.ndarray, optional
    :param figsize: Figure size, used only when a new figure is created (``axes=None``).
    :type figsize: tuple, optional, default=(15, 4)
    :param axes: Optional sequence of exactly 3 existing axes to draw the
        (rate-time, scale-time, frequency-time) panels into -- e.g. one row of a
        larger subplot grid, so several representations can be compared in a single figure.
        If None, a new 1x3 figure is created.
    :type axes: sequence of matplotlib.axes.Axes, optional

    :returns: The figure and its three axes ``(fig, (ax_rate, ax_scale, ax_freq))``.
    :rtype: tuple
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")

    n_rate = len(rates)

    if cr.shape[1] == 2 * n_rate:
        # both sweep directions present: magnitude of the analytic output,
        # averaged over the two sweep directions
        mag = (np.abs(cr[:, :n_rate, :, :]) + np.abs(cr[:, n_rate:2*n_rate, :, :])) / 2  # [scale, rate, time, freq]
    elif cr.shape[1] == n_rate:
        # single direction already selected by the caller: use as-is, no averaging
        mag = np.abs(cr)  # [scale, rate, time, freq]
    else:
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal either len(rates) ({n_rate}), "
            f"for a single sweep direction, or 2*len(rates) ({2*n_rate}), for both directions."
        )

    rate_time = mag.mean(axis=(0, 3))          # [rate, time]
    scale_time = mag.mean(axis=(1, 3))         # [scale, time]
    freq_time = mag.mean(axis=(0, 1)).T        # [freq, time]

    n_time_actual = mag.shape[2]

    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        axes = np.atleast_1d(axes).ravel()
        if axes.size != 3:
            raise ValueError("axes must contain exactly 3 Axes "
                             "(rate-time, scale-time, frequency-time).")
        fig = axes[0].figure

    plots = [
        {"ax": axes[0], "data": rate_time, "ylabel": "Rate [Hz]", "title": "Rate-Time"},
        {"ax": axes[1], "data": scale_time, "ylabel": "Scale [cyc/oct]", "title": "Scale-Time"},
        {"ax": axes[2], "data": freq_time, "ylabel": "Frequency [Hz]", "title": "Frequency-Time"},
    ]

    for p in plots:
        ax = p["ax"]
        im = ax.imshow(p["data"], aspect='auto', origin='lower', cmap="viridis")
        ax.set_xlabel("Time [s]" if time_points is not None else "Time [frame]")
        ax.set_ylabel(p["ylabel"])
        ax.set_title(p["title"])
        fig.colorbar(im, ax=ax)

    ax_rate, ax_scale, ax_freq = axes

    # Time ticks (x-axis on all panels), offset into the real-data region if margins exist
    if time_points is not None:
        dN = (n_time_actual - len(time_points)) // 2
        if dN < 0:
            raise ValueError(
                f"len(time_points) ({len(time_points)}) exceeds cr's time dimension "
                f"({n_time_actual}); time_points must correspond to the unpadded time axis."
            )
        t_idx = np.linspace(0, len(time_points) - 1, 6).astype(int)
        t_ticks = t_idx + dN
        t_labels = [f"{time_points[i]:.2f}" for i in t_idx]
        for ax in axes:
            ax.set_xticks(t_ticks)
            ax.set_xticklabels(t_labels, rotation=45)

    # Rate ticks (y-axis on ax_rate; rates are not padded)
    r_idx = np.linspace(0, n_rate - 1, 6).astype(int)
    ax_rate.set_yticks(r_idx)
    ax_rate.set_yticklabels([f"{rates[i]:.1f}" for i in r_idx])

    # Scale ticks (y-axis on ax_scale; scales are not padded)
    if scales is not None:
        s_idx = np.linspace(0, len(scales) - 1, 6).astype(int)
        ax_scale.set_yticks(s_idx)
        ax_scale.set_yticklabels([f"{scales[i]:.1f}" for i in s_idx])

    # Frequency ticks (y-axis on ax_freq), offset into the real-data region
    if frequencies is not None:
        n_freq_actual = freq_time.shape[0]
        dM = (n_freq_actual - len(frequencies)) // 2
        if dM < 0:
            raise ValueError(
                f"len(frequencies) ({len(frequencies)}) exceeds cr's frequency dimension "
                f"({n_freq_actual}); frequencies must correspond to the unpadded axis."
            )
        f_idx = np.linspace(0, len(frequencies) - 1, 6).astype(int)
        ax_freq.set_yticks(f_idx + dM)
        ax_freq.set_yticklabels([f"{frequencies[i]:.0f}" for i in f_idx])

    if created:
        fig.tight_layout()
    return fig, (ax_rate, ax_scale, ax_freq)


def save_wav(signal: np.ndarray, sf: int, filepath: Path) -> None:
    """
    Saves a 1-D signal as a .wav file, normalized to 16-bit PCM (pulse code modulation) range.

    :param signal: 1-D array of audio samples.
    :type signal: numpy.ndarray

    :param sf: Sampling frequency, in Hz.
    :type sf: int

    :param filepath: Path to save the .wav file to, including filename and extension.
    :type filepath: pathlib.Path

    :returns: None. Writes the file to disk.
    :rtype: None
    """
    normalized = signal / np.max(np.abs(signal))
    scaled = (normalized * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(filepath, sf, scaled)

    print(f"Successfully saved .wav file to {filepath}")

    return

def plot_tempfilt_response(H, fps, center=None, max_freq=None,
                           title=None, ax=None):
    """Plot the magnitude response of a single temporal filter.

    :param H: Frequency response returned by ``gen_cort``.
    :type H: numpy.ndarray
    :param fps: Frame rate (frames per second) used to generate the filter.
    :type fps: float
    :param center: Optional center frequency to mark with a vertical line.
    :type center: float, optional
    :param max_freq: If given, limit the displayed frequency axis.
    :type max_freq: float, optional
    :param title: Optional plot title.
    :type title: str, optional
    :param ax: Existing matplotlib axes to draw on. A new figure/axes is
        created if omitted.
    :type ax: matplotlib.axes.Axes, optional

    :returns: The axes the filter response was drawn on.
    :rtype: matplotlib.axes.Axes
    """

    freqs = np.fft.fftfreq(2 * len(H), d=1 / fps)[:len(H)]

    if max_freq is not None:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        H = H[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.plot(freqs, np.abs(H), lw=1)

    if center is not None:
        ax.axvline(center, color="k", linestyle="--", linewidth=1,
                   label=f"Center = {center:g} Hz")
        ax.legend()

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain")

    if title is not None:
        ax.set_title(title)

    ax.grid(True)

    return ax

def plot_spectfilt_response(H, ch_per_oct, max_scale=None, title=None, ax=None):
    """Plot the magnitude response of a single cortical scale (spectral)
    filter, as produced by one call to ``gen_corf``.
 
    :param H: The filter's magnitude response, i.e. the array returned by
        ``gen_corf(fc, L, ch_per_oct, func_type)``. Its length is used directly
        to build the frequency axis, so pass in ``H`` exactly as returned.
    :type H: numpy.ndarray
    :param ch_per_oct: Channels per octave used when ``H`` was generated
        (the same ``ch_per_oct`` / ``SRF`` argument passed to ``gen_corf``).
        Needed here to convert array index into cycles/octave.
    :type ch_per_oct: int
    :param max_scale: If given, the x-axis is cut off at this scale
        (cycles/octave) for readability. This only changes the view, not
        the data -- the full ``H`` is still plotted underneath.
    :type max_scale: float, optional
    :param title: Optional title for the plot.
    :type title: str, optional
    :param ax: Existing matplotlib axes to draw on. A new figure/axes is
        created if this is omitted.
    :type ax: matplotlib.axes.Axes, optional
 
    :returns: The axes the filter response was drawn on.
    :rtype: matplotlib.axes.Axes
 
    .. note:: ``gen_corf`` returns a purely real magnitude array (no phase
        term, unlike ``gen_cort``)
    """
    H = np.asarray(H)
    L = len(H)
 
    # Real frequency axis in cyc/oct: index m of H corresponds to
    # m/L * ch_per_oct/2 (same derivation as R1 inside gen_corf, just
    # without dividing out fc).
    freqs = np.arange(L) / L * ch_per_oct / 2
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
 
    ax.plot(freqs, H)
    ax.set_xlabel("Scale (cycles/octave)")
    ax.set_ylabel("Magnitude")
 
    if title is not None:
        ax.set_title(title)
 
    if max_scale is not None:
        ax.set_xlim(0, max_scale)

    ax.grid(True)
    return ax