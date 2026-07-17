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

    :param matrix: 2-D array of magnitude values to plot, of shape (len(frequencies), len(time_points)).
        Values are converted to decibels internally; values less than or equal to zero are clipped to
        a small positive constant beforehand to avoid taking the log of zero.
    :type matrix: numpy.ndarray

    :param time_points: 1-D array of time values (in seconds) corresponding to the columns of ``matrix``.
    :type time_points: numpy.ndarray

    :param frequencies: 1-D array of frequency values (in Hz) corresponding to the rows of ``matrix``.
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
    plt.pcolormesh(time_points, frequencies, to_decibel(matrix))
    plt.colorbar(label='Magnitude (dB)')

    plt.xlabel("Time (s)")
    plt.yscale('log', base=2)
    plt.ylim(*ylim)
    plt.ylabel("Frequency (Hz)")
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

def plot_cr_projection(cr, rates, scales=None, frequencies=None,
                        time_margin=None, cmap="viridis", figsize=(12, 4)):
    """
    Plot 2D projections (scale-rate, scale-frequency, rate-frequency)
    of a cortical representation.

    cr : np.ndarray
        4-D cortical output, shape (num_scales, num_rates*2, num_time, num_freq).
    rates : array-like
        Rate vector used in aud2cor (length = num_rates, i.e. half of cr.shape[1]).
    scales : array-like, optional
        Scale vector used in aud2cor (for real tick labels). If None, axis shows index.
    frequencies : array-like, optional
        Characteristic frequencies (for real tick labels). If None, axis shows index.
    time_margin : int, optional
        If you used tp_margin > 0 in aud2cor, pass the number of margin samples
        to crop from each end of the time axis before averaging (mirrors the
        `500:end-500` cropping in the original MATLAB script).
    """
    if cr.ndim != 4:
        raise ValueError("cr must be a 4-D array with shape (scale, rate, time, frequency).")
    if cr.shape[1] != 2 * len(rates):
        raise ValueError(
            f"cr.shape[1] ({cr.shape[1]}) must equal 2*len(rates) ({2*len(rates)})."
        )

    n_rate = len(rates)

    # optionally crop time margins before averaging
    if time_margin:
        t_slice = slice(time_margin, cr.shape[2] - time_margin)
    else:
        t_slice = slice(None)

    cr_up = np.mean(np.abs(cr[:, :n_rate, t_slice, :]), axis=2)
    cr_down = np.mean(np.abs(cr[:, n_rate:2*n_rate, t_slice, :]), axis=2)
    cr_avgr = (cr_up + cr_down) / 2  # [scale x rate x frequency]

    scale_rate = np.mean(cr_avgr, axis=2)   # [scale x rate]
    scale_freq = np.mean(cr_avgr, axis=1)   # [scale x frequency]
    rate_freq = np.mean(cr_avgr, axis=0)    # [rate x frequency]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    im1 = ax1.imshow(scale_rate, aspect='auto', origin='lower', cmap=cmap)
    ax1.set_xlabel('Rate [Hz]'); ax1.set_ylabel('Scale [cyc/oct]')
    ax1.set_title('Scale-Rate')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(scale_freq, aspect='auto', origin='lower', cmap=cmap)
    ax2.set_xlabel('Frequency [Hz]'); ax2.set_ylabel('Scale [cyc/oct]')
    ax2.set_title('Scale-Frequency')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(rate_freq, aspect='auto', origin='lower', cmap=cmap)
    ax3.set_xlabel('Frequency [Hz]'); ax3.set_ylabel('Rate [Hz]')
    ax3.set_title('Rate-Frequency')
    plt.colorbar(im3, ax=ax3)

    # optional: real tick labels instead of array indices
    if frequencies is not None:
        f_idx = np.linspace(0, len(frequencies)-1, 6).astype(int)
        for ax in (ax2, ax3):
            ax.set_xticks(f_idx)
            ax.set_xticklabels([f"{frequencies[i]:.0f}" for i in f_idx], rotation=45)
    if rates is not None:
        r_idx = np.linspace(0, len(rates)-1, 6).astype(int)
        for ax in (ax1, ax3):
            ax.set_yticks(r_idx) if ax is ax3 else ax.set_xticks(r_idx)
            labels = [f"{rates[i]:.1f}" for i in r_idx]
            ax.set_yticklabels(labels) if ax is ax3 else ax.set_xticklabels(labels)
    if scales is not None:
        s_idx = np.linspace(0, len(scales)-1, 6).astype(int)
        for ax in (ax1, ax2):
            ax.set_yticks(s_idx)
            ax.set_yticklabels([f"{scales[i]:.1f}" for i in s_idx])

    plt.tight_layout()
    return fig, (ax1, ax2, ax3)

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