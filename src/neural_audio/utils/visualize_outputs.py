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

    This function draws onto the *current* matplotlib axes and does not create a new figure or call
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


def plot_cr_projection(cr, rates, scales=None, frequencies=None,
                        cmap="viridis", figsize=(12, 4)):
    """
    Plot 2D projections (scale-rate, scale-frequency, rate-frequency)
    of a cortical representation.

    cr : np.ndarray
        4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
        num_time and num_freq may include zero/wrap-around margins added by
        aud2cor's tp_margin / sp_margin (no cropping is done here; margin
        columns/rows are shown but left unlabeled).
    rates : array-like
        Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    scales : array-like, optional
        Scale vector used in aud2cor (for real tick labels). If None, axis shows index.
    frequencies : array-like, optional
        Characteristic frequencies from wav2aud (for real tick labels). If None,
        axis shows index. Length is expected to be <= cr.shape[3]; if cr's frequency
        axis is wider (because sp_margin > 0 was used in aud2cor), the extra margin
        columns are included in the plot but left unlabeled.
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")
    if cr.shape[1] != 2 * len(rates):
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal 2*len(rates) ({2*len(rates)})."
        )

    n_rate = len(rates)

    cr_up = np.mean(np.abs(cr[:, :n_rate, :, :]), axis=2)
    cr_down = np.mean(np.abs(cr[:, n_rate:2*n_rate, :, :]), axis=2)
    cr_avgr = (cr_up + cr_down) / 2  # [scale x rate x frequency]

    scale_rate = np.mean(cr_avgr, axis=2)   # [scale x rate]
    scale_freq = np.mean(cr_avgr, axis=1)   # [scale x frequency]
    rate_freq = np.mean(cr_avgr, axis=0)    # [rate x frequency]

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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plots = [
        {"ax": axes[0], "data": scale_rate, "xlabel": "Rate [Hz]",
         "ylabel": "Scale [cyc/oct]", "title": "Scale-Rate"},
        {"ax": axes[1], "data": scale_freq, "xlabel": "Frequency [Hz]",
         "ylabel": "Scale [cyc/oct]", "title": "Scale-Frequency"},
        {"ax": axes[2], "data": rate_freq, "xlabel": "Frequency [Hz]",
         "ylabel": "Rate [Hz]", "title": "Rate-Frequency"},
    ]

    for p in plots:
        ax = p["ax"]
        im = ax.imshow(p["data"], aspect='auto', origin='lower', cmap=cmap)
        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel(p["ylabel"])
        ax.set_title(p["title"])
        plt.colorbar(im, ax=ax)

    ax1, ax2, ax3 = axes

    # Frequency ticks (x-axis on ax2, ax3), offset into the real-data region
    if frequencies is not None:
        f_idx = np.linspace(0, len(frequencies) - 1, 6).astype(int)
        f_ticks = f_idx + dM
        f_labels = [f"{frequencies[i]:.0f}" for i in f_idx]
        for ax in (ax2, ax3):
            ax.set_xticks(f_ticks)
            ax.set_xticklabels(f_labels, rotation=45)

    # Rate ticks: x-axis on ax1, y-axis on ax3 (no margin, rates aren't padded)
    if rates is not None:
        r_idx = np.linspace(0, len(rates) - 1, 6).astype(int)
        r_labels = [f"{rates[i]:.1f}" for i in r_idx]
        ax1.set_xticks(r_idx)
        ax1.set_xticklabels(r_labels)
        ax3.set_yticks(r_idx)
        ax3.set_yticklabels(r_labels)

    # Scale ticks: y-axis on ax1, ax2 (no margin, scales aren't padded)
    if scales is not None:
        s_idx = np.linspace(0, len(scales) - 1, 6).astype(int)
        s_labels = [f"{scales[i]:.1f}" for i in s_idx]
        for ax in (ax1, ax2):
            ax.set_yticks(s_idx)
            ax.set_yticklabels(s_labels)

    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_cr_temporal(cr, rates, scales=None, frequencies=None, time_points=None,
                     cmap="viridis", figsize=(15, 4)):
    """Plot time-resolved projections (rate-time, scale-time, frequency-time)
    of a cortical representation.

    :func:`plot_cr_projection` collapses the time axis and therefore shows only the
    *time-averaged* modulation content, hiding how the cortical energy evolves over the
    course of the stimulus. This function instead keeps **time on the horizontal axis**
    and collapses the two axes that are not being examined, giving three complementary
    time-resolved views:

    - **Rate-Time**: how temporal-modulation (rate) energy changes over time
      (averaged over scale and frequency).
    - **Scale-Time**: how spectral-modulation (scale) energy changes over time
      (averaged over rate and frequency).
    - **Frequency-Time**: how per-channel energy changes over time
      (averaged over scale and rate).

    Design choices:

    - The **magnitude** ``|cr|`` is used. ``aud2cor`` returns a complex (analytic) output;
      its magnitude is the modulation *envelope*, which is the interpretable, non-oscillating
      quantity to track over time (the phase is discussed in the tutorial).
    - The two sweep directions (upward/downward) are **averaged** together, matching
      :func:`plot_cr_projection`, so the two functions can be read side by side. Pass a
      single-direction slice of ``cr`` if you want to inspect one direction on its own.

    :param cr: 4-D cortical output, shape ``(num_scales, num_rates*2, num_time, num_freq)``.
        ``num_time`` / ``num_freq`` may include the margins added by ``aud2cor``'s
        ``tp_margin`` / ``sp_margin``; margin rows/columns are shown but left unlabeled.
    :type cr: numpy.ndarray
    :param rates: Rate vector used in ``aud2cor`` (length ``= num_rates``, i.e. half of
        ``cr.shape[1]``).
    :type rates: array-like
    :param scales: Scale vector used in ``aud2cor`` (for real y-tick labels on the
        Scale-Time panel). If ``None``, the axis shows the channel index.
    :type scales: array-like, optional
    :param frequencies: Characteristic frequencies from ``wav2aud`` (for real y-tick labels
        on the Frequency-Time panel). If ``None``, the axis shows the channel index.
    :type frequencies: array-like, optional
    :param time_points: Time values (in seconds) for the *unpadded* time frames, e.g. the
        ``time_points`` returned by ``wav2aud``. Used for real x-tick labels; when shorter
        than ``cr``'s time axis (because ``tp_margin > 0`` was used) the labels are offset
        into the real-data region. If ``None``, the axis shows the frame index.
    :type time_points: array-like, optional
    :param cmap: Matplotlib colormap name.
    :type cmap: str, optional, default='viridis'
    :param figsize: Figure size passed to ``plt.subplots``.
    :type figsize: tuple, optional, default=(15, 4)

    :returns: The created figure and its three axes ``(fig, (ax_rate, ax_scale, ax_freq))``.
    :rtype: tuple
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")
    if cr.shape[1] != 2 * len(rates):
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal 2*len(rates) ({2*len(rates)})."
        )

    n_rate = len(rates)

    # magnitude of the analytic output, averaged over the two sweep directions
    mag = (np.abs(cr[:, :n_rate, :, :]) + np.abs(cr[:, n_rate:2*n_rate, :, :])) / 2  # [scale, rate, time, freq]

    rate_time = mag.mean(axis=(0, 3))          # [rate, time]
    scale_time = mag.mean(axis=(1, 3))         # [scale, time]
    freq_time = mag.mean(axis=(0, 1)).T        # [freq, time]

    n_time_actual = mag.shape[2]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plots = [
        {"ax": axes[0], "data": rate_time, "ylabel": "Rate [Hz]", "title": "Rate-Time"},
        {"ax": axes[1], "data": scale_time, "ylabel": "Scale [cyc/oct]", "title": "Scale-Time"},
        {"ax": axes[2], "data": freq_time, "ylabel": "Frequency [Hz]", "title": "Frequency-Time"},
    ]

    for p in plots:
        ax = p["ax"]
        im = ax.imshow(p["data"], aspect='auto', origin='lower', cmap=cmap)
        ax.set_xlabel("Time [s]" if time_points is not None else "Time [frame]")
        ax.set_ylabel(p["ylabel"])
        ax.set_title(p["title"])
        plt.colorbar(im, ax=ax)

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

    plt.tight_layout()
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