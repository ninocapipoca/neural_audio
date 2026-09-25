import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections.abc import Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from scipy.io import wavfile

def to_decibel(x: np.ndarray, vmin: float | None = None, normalize: bool = False) -> np.ndarray:
    """Convert an array of magnitudes to decibels.

    The decibel scale is referenced to a magnitude of 1.0, so magnitudes below
    1.0 come out negative. Pass ``normalize=True`` for the peak-relative
    scale instead.

    :param x: Array of magnitudes. Values at or below zero are clipped to a small positive
        constant (1e-9) beforehand to avoid taking the log of zero.
    :type x: numpy.ndarray

    :param vmin: Floor, in decibels. Values below it are clipped up to it. Useful because the
        1e-9 clip maps exact zeros to -180 dB, which stretches a colour scale over a range far
        wider than the content occupies. ``None`` applies no floor.
    :type vmin: float, optional, default=None

    :param normalize: If ``True``, express the magnitudes relative to the largest value in
        ``x``, so that the peak becomes 0 dB and every other value is negative.
    :type normalize: bool, optional, default=False

    :returns: The input converted to decibels, same shape as ``x``.
    :rtype: numpy.ndarray
    """

    magnitude = np.maximum(x, 1e-9)

    if normalize:
        magnitude = magnitude / magnitude.max()

    decibels = 20 * np.log10(magnitude)

    if vmin is not None:
        decibels = np.maximum(decibels, vmin)

    return decibels


def plot_spectrogram(matrix: np.ndarray,
                      time_points: np.ndarray,
                      frequencies: np.ndarray,
                      title: str='Spectrogram',
                      vmin: float | None = None,
                      normalize: bool = False) -> None:
    
    """Plots a spectrogram-like matrix on a log-scaled frequency axis, with magnitude converted to decibels. Designed
    to work for both a regular spectrogram and for an audiogram as produced by ``wav2aud``.

    The frequency axis is displayed on a base-2 log scale so that octave spacing appears linear.
    ``pcolormesh`` is used rather than ``imshow`` so that linearly-spaced FFT bins
    can be mapped onto this log-scaled axis correctly.

    This function draws onto the current matplotlib axes and does not create a new figure or call
    ``plt.show()`` itself, so it can be used with ``plt.subplot`` to place multiple spectrograms side
    by side or stacked.

    .. note:: The argument ``matrix`` is expected to be of shape ``[timepoints, frequencies]``, matching
        the shape of ``wav2aud``'s output. A regular spectrogram, for example as returned by ``scipy.signal.spectrogram``,
        has the opposite arrangement, so its transpose will need to be passed instead.

    Example usage:
    .. code-block:: python
    
            plt.figure(figsize=(12, 6))
    
            plt.subplot(2, 1, 1)
            plot_spectrogram(matrix=spectrogram.T, time_points=time_points_spectrogram,
                              frequencies=frequencies_spectrogram,
                              title="Regular Spectrogram")
    
            plt.subplot(2, 1, 2)
            plot_spectrogram(matrix=audiogram, time_points=time_points_audiogram,
                              frequencies=frequencies_audiogram,
                              title="Audiogram")
    
            plt.tight_layout()
            plt.show()

    :param matrix: 2-D array of magnitude values to plot, of shape ``[timepoints, frequencies]``.
         Values are converted to decibels internally by ``to_decibel``.
    :type matrix: numpy.ndarray

    :param time_points: 1-D array of time values (in seconds) corresponding to the rows of ``matrix``.
    :type time_points: numpy.ndarray

    :param frequencies: 1-D array of frequency values (in Hz) corresponding to the columns of ``matrix``.
    :type frequencies: numpy.ndarray

    :param title: Title displayed above the plot.
    :type title: str, optional, default='Spectrogram'

    :param vmin: Decibel floor passed to ``to_decibel``, clipping the low end of the colour scale.
    :type vmin: float, optional, default=None

    :param normalize: If ``True``, plot decibels relative to the peak of ``matrix`` rather than
        to an absolute magnitude of 1.0. Passed to ``to_decibel``.
    :type normalize: bool, optional, default=False

    :returns: None.
    :rtype: None
    """

    # Due to log scale, frequency axis cannot show 0 Hz, fall back to lowest positive frequency in that case.
    positive_frequencies = frequencies[frequencies > 0]
    ylim = [positive_frequencies[0], positive_frequencies[-1]] if positive_frequencies.size \
        else [frequencies[0], frequencies[-1]]

    plt.title(title)

    plt.pcolormesh(time_points, frequencies, to_decibel(matrix, vmin=vmin, normalize=normalize).T)
    plt.colorbar(label='Magnitude (dB)')

    plt.xlabel("Time (s)")
    plt.yscale('log', base=2)
    plt.ylim(*ylim)
    plt.ylabel("Frequency (Hz)")
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())


def cr_projections(cr: np.ndarray, rates: np.ndarray,
                   signed_rates: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three 2-D projections of a cortical representation.

    The magnitude ``|cr|`` is reduced over the time axis and then collapsed onto each
    remaining pair of axes. This is the shared computation behind
    :func:`plot_cr_projection`; it is exposed separately so a cortical representation can
    be reduced to its projections without plotting.

    ``cr``'s rate axis has length ``2*len(rates)``: the first half is the ``sgn = -1``
    sweep-direction and the second half the ``sgn = +1`` direction (see
    :func:`~neural_audio.aud2cor.aud2cor`). ``signed_rates`` controls how the two halves
    are combined:

    - ``False`` (default): the two directions are averaged together, so the rate axis
      of the returned projections has length ``len(rates)`` and direction is discarded.
    - ``True``: the two directions are kept side by side on a single signed rate axis
      of length ``2*len(rates)``. The first half is reversed onto negative rates and the
      second half onto positive rates, matching
      ``np.concatenate([-rates[::-1], rates])``. ``scale_rate`` and ``rate_freq`` then
      carry the full ``2*len(rates)`` rate axis; ``scale_freq`` is identical in both modes.


    :param cr: 4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
    :type cr: numpy.ndarray
    :param rates: Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    :type rates: numpy.ndarray
    :param signed_rates: If ``True``, keep both sweep directions on a signed rate axis
        instead of averaging them. See above.
    :type signed_rates: bool, optional, default=False

    :returns: Tuple ``(scale_rate, scale_freq, rate_freq)`` of 2-D arrays.
    :rtype: tuple
    """
    n_rate = len(rates)

    # Magnitude, time-averaged scale-by-scale to reduce memory usage
    mag = np.empty(cr.shape[:2] + cr.shape[3:])
    for s in range(cr.shape[0]):
        mag[s] = np.abs(cr[s]).mean(axis=1)

    if signed_rates:
        # first half reversed onto negative rates, second half onto positive
        order = np.concatenate([np.arange(n_rate)[::-1], np.arange(n_rate) + n_rate])
        reduced = mag[:, order]                                  # [scale, 2*n_rate, freq]
    else:
        # sweep directions averaged -> [scale, n_rate, freq]
        reduced = (mag[:, :n_rate] + mag[:, n_rate:2 * n_rate]) / 2

    return reduced.mean(2), reduced.mean(1), reduced.mean(0)


def plot_cr_projection(cr: np.ndarray, rates: np.ndarray, scales: np.ndarray | None = None,
                       frequencies: np.ndarray | None = None, figsize: tuple[float, float] = (12, 4),
                       axes: Sequence[Axes] | None = None,
                       signed_rates: bool = False) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plots 2D projections (scale-rate, scale-frequency, rate-frequency) of a cortical representation produced by `aud2cor`, with time averaged out.

    :param cr: 4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
        num_time and num_freq may include zero/wrap-around margins added by aud2cor's
        temporal_margin / spectral_margin (note that margin columns/rows are shown
        but left unlabeled since they have no physical meaning).
    :type cr: numpy.ndarray
    :param rates: Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    :type rates: numpy.ndarray
    :param scales: Scale vector used in aud2cor (for real tick labels). If None, the axis
        shows the index.
    :type scales: numpy.ndarray, optional
    :param frequencies: Characteristic frequencies from wav2aud (for real tick labels). If
        None, the axis shows the index. Length is expected to be <= cr.shape[3]; if cr's
        frequency axis is wider (because spectral_margin > 0 was used in aud2cor), the extra
        margin columns are included in the plot but left unlabeled.
    :type frequencies: numpy.ndarray, optional
    :param figsize: Figure size, used only when a new figure is created (``axes=None``).
    :type figsize: tuple of float, optional, default=(12, 4)
    :param axes: Optional sequence of exactly 3 existing axes to draw the
        (scale-rate, scale-frequency, rate-frequency) panels into -- e.g. one row of a
        larger subplot grid, so several representations can be compared in a single figure.
        If None, a new 1x3 figure is created.
    :type axes: list of matplotlib.axes.Axes, optional
    :param signed_rates: If ``True``, keep the two sweep directions on a single signed rate
        axis (running ``-rates[::-1] .. +rates``) instead of the default behavior of averaging them together. The
        Scale-Rate and Rate-Frequency panels then span both directions, with a dashed line
        marking the boundary between the negative and positive halves and the peak landing
        on the side matching the stimulus' sweep direction. Both modes share a single colour scale per
        panel, so the weaker direction is not brightened to match the stronger one.
    :type signed_rates: bool, optional, default=False

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

    scale_rate, scale_freq, rate_freq = cr_projections(cr, rates, signed_rates=signed_rates)

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

    # Rate ticks: x-axis on ax1, y-axis on ax3 (rates aren't padded, so no margin offset)
    if signed_rates:
        # signed axis of length 2*len(rates): -rates[::-1] .. +rates
        signed = np.concatenate([-rates[::-1], rates])
        r_idx = np.linspace(0, len(signed) - 1, 6).astype(int)
        r_labels = [f"{signed[i]:+.1f}" for i in r_idx]
        # dashed divider between the negative and positive halves
        boundary = len(rates) - 0.5
        ax1.axvline(boundary, color="grey", ls="--", lw=1)
        ax3.axhline(boundary, color="grey", ls="--", lw=1)
    else:
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


def plot_cr_temporal(cr: np.ndarray, rates: np.ndarray, scales: np.ndarray | None = None,
                     time_points: np.ndarray | None = None,
                     figsize: tuple[float, float] = (10, 4),
                     axes: Sequence[Axes] | None = None,
                     signed_rates: bool = False) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plots time-resolved projections (rate-time, scale-time) of a cortical representation produced by `aud2cor`.

    Rather than collapsing the time axis as in :func:`plot_cr_projection`, this function instead keeps time on the x-axis
    and collapses the two axes that are not being examined, producing two time-resolved views:

    - Rate-Time: how temporal-modulation (rate) energy changes over time
      (averaged over scale and frequency).
    - Scale-Time: how spectral-modulation (scale) energy changes over time
      (averaged over rate and frequency).

    The magnitude ``|cr|`` is used, since it acts as a strength-of-match measure for a pair
    (scale, rate), essentially quantifying how much the signal is acting like (scale, rate).

    ``cr`` contains both sweep directions (``cr.shape[1] == 2*len(rates)``). By default the two
    directions are averaged together, as in :func:`plot_cr_projection`. Pass ``signed_rates=True``
    to keep them apart on a signed rate axis instead (see below).

    :param cr: 4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
        ``num_time`` may include the margins added by ``aud2cor``'s
        ``temporal_margin``; margin columns are shown but left unlabeled.
    :type cr: numpy.ndarray
    :param rates: Rate vector used in ``aud2cor`` (length ``= num_rates``).
    :type rates: numpy.ndarray
    :param scales: Scale vector used in ``aud2cor`` (for real y-tick labels on the
        Scale-Time panel). If ``None``, the axis shows the channel index.
    :type scales: numpy.ndarray, optional
    :param time_points: Time values (in seconds) for the *unpadded* time frames, e.g. the
        ``time_points`` returned by ``wav2aud``. Used for real x-tick labels; when shorter
        than ``cr``'s time axis (because ``temporal_margin > 0`` was used) the labels are offset
        into the real-data region. If ``None``, the axis shows the frame index.
    :type time_points: numpy.ndarray, optional
    :param figsize: Figure size, used only when a new figure is created (``axes=None``).
    :type figsize: tuple of float, optional, default=(10, 4)
    :param axes: Optional sequence of exactly 2 existing axes to draw the
        (rate-time, scale-time) panels into -- e.g. one row of a
        larger subplot grid, so several representations can be compared in a single figure.
        If None, a new 1x2 figure is created.
    :type axes: list of matplotlib.axes.Axes, optional
    :param signed_rates: If ``True``, keep the two sweep directions on a single signed rate
        axis (running ``-rates[::-1] .. +rates``) instead of the default behavior of averaging
        them together. The Rate-Time panel then spans both directions, with a dashed line
        marking the boundary between the negative and positive halves and the energy landing
        on the side matching the stimulus' sweep direction. The Scale-Time panel is the same in
        both modes. Both modes share a single colour scale per panel, so the weaker direction
        is not brightened to match the stronger one.
    :type signed_rates: bool, optional, default=False

    :returns: The figure and its two axes ``(fig, (ax_rate, ax_scale))``.
    :rtype: tuple
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")
    if cr.shape[1] != 2 * len(rates):
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal 2*len(rates) ({2*len(rates)})."
        )

    n_rate = len(rates)

    if signed_rates:
        # first half reversed onto negative rates, second half onto positive
        order = np.concatenate([np.arange(n_rate)[::-1], np.arange(n_rate) + n_rate])
        mag = np.abs(cr[:, order])  # [scale, 2*rate, time, freq]
    else:
        # magnitude of the analytic output, averaged over the two sweep directions
        mag = (np.abs(cr[:, :n_rate, :, :]) + np.abs(cr[:, n_rate:2*n_rate, :, :])) / 2  # [scale, rate, time, freq]

    rate_time = mag.mean(axis=(0, 3))          # [rate, time]
    scale_time = mag.mean(axis=(1, 3))         # [scale, time]

    n_time_actual = mag.shape[2]

    created = axes is None
    if created:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.atleast_1d(axes).ravel()
        if axes.size != 2:
            raise ValueError("axes must contain exactly 2 Axes "
                             "(rate-time, scale-time).")
        fig = axes[0].figure

    plots = [
        {"ax": axes[0], "data": rate_time, "ylabel": "Rate [Hz]", "title": "Rate-Time"},
        {"ax": axes[1], "data": scale_time, "ylabel": "Scale [cyc/oct]", "title": "Scale-Time"},
    ]

    for p in plots:
        ax = p["ax"]
        im = ax.imshow(p["data"], aspect='auto', origin='lower', cmap="viridis")
        ax.set_xlabel("Time [s]" if time_points is not None else "Time [frame]")
        ax.set_ylabel(p["ylabel"])
        ax.set_title(p["title"])
        fig.colorbar(im, ax=ax)

    ax_rate, ax_scale = axes

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
    if signed_rates:
        # signed axis of length 2*len(rates): -rates[::-1] .. +rates
        signed = np.concatenate([-rates[::-1], rates])
        r_idx = np.linspace(0, len(signed) - 1, 6).astype(int)
        r_labels = [f"{signed[i]:+.1f}" for i in r_idx]
        # dashed divider between the negative and positive halves
        ax_rate.axhline(n_rate - 0.5, color="grey", ls="--", lw=1)
    else:
        r_idx = np.linspace(0, n_rate - 1, 6).astype(int)
        r_labels = [f"{rates[i]:.1f}" for i in r_idx]
    ax_rate.set_yticks(r_idx)
    ax_rate.set_yticklabels(r_labels)

    # Scale ticks (y-axis on ax_scale; scales are not padded)
    if scales is not None:
        s_idx = np.linspace(0, len(scales) - 1, 6).astype(int)
        ax_scale.set_yticks(s_idx)
        ax_scale.set_yticklabels([f"{scales[i]:.1f}" for i in s_idx])

    if created:
        fig.tight_layout()
    return fig, (ax_rate, ax_scale)


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

def plot_tempfilt_response(H: np.ndarray, fps: float, center: float | None = None,
                           max_freq: float | None = None, title: str | None = None,
                           ax: Axes | None = None) -> Axes:
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

def plot_spectfilt_response(H: np.ndarray, channels_per_oct: int, max_scale: float | None = None,
                            title: str | None = None, ax: Axes | None = None) -> Axes:
    """Plot the magnitude response of a single cortical scale (spectral)
    filter, as produced by one call to ``gen_corf``.
 
    :param H: The filter's magnitude response, i.e. the array returned by
        ``gen_corf(fc, L, channels_per_oct, func_type)``. Its length is used directly
        to build the frequency axis, so pass in ``H`` exactly as returned.
    :type H: numpy.ndarray
    :param channels_per_oct: Channels per octave used when ``H`` was generated
        (the same ``channels_per_oct`` / ``SRF`` argument passed to ``gen_corf``).
        Needed here to convert array index into cycles/octave.
    :type channels_per_oct: int
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
     """
    H = np.asarray(H)
    L = len(H)
 
    # Real frequency axis in cyc/oct: index m of H corresponds to
    # m/L * channels_per_oct/2 (same derivation as R1 inside gen_corf, just
    # without dividing out fc).
    freqs = np.arange(L) / L * channels_per_oct / 2
 
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