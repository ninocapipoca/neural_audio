import numpy as np
from scipy import signal as sig
from neural_audio.utils import mathfuncs as mf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import logging
import os

logger = logging.getLogger(__name__)

# Number of output channels processed per _wav2aud_step call in the serial path (bounds memory use)
_SERIAL_BLOCK = 32

def wav2aud(x: np.ndarray,
            octave_shift: int=0,
            frame_length: int=4,
            sigmoid_factor: float=-2,
            time_constant: int=0,
            verbose: bool=False,
            filters: dict = dict(),
            *,
            parallel: bool = False,
            n_jobs: int | None = None,
            backend: str = "threads",
            chunk_size: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    r"""This function computes a biologically inspired spectrogram (also known as audiogram) of an acoustic waveform by simulating the
    transduction mechanism of the ferret cochlea, following the NSL Matlab toolbox by Yang, Wang and Shamma (1992 [1]_ and 1994 [2]_).
    
    A section of the cochlea spanning 128 hair-cells with a frequency selectivity calibrated to the range 180-7040Hz (given
    ``octave_shift=0``) are each simulated by:

        * convolving the signal with a corresponding cell-specific bandpass filter that is implemented as a second-order section infinite impulse response (IIR) function.
        * applying a non-linear activation function to the output of each filter, modeling the hair cell transduction. The nonlinearity can be a sigmoid, step function, rectified linear function, or identity function, depending on the value of ``sigmoid_factor``.
        * applying a low-pass filter to the output of each nonlinearity, modeling the hair cell membrane potential. This step is skipped when ``sigmoid_factor`` is set to -2, which models a linear ionic channel mode without membrane potential dynamics.
        * computing lateral inhibition between adjacent hair-cells by subtracting from each channel the output of the channel above it (next higher frequency), followed by half-wave rectification. Note, to compute lateral inhibition for the channel of highest frequency, an auxiliary hair-cell with higher frequency selectivity is used. This auxiliary hair-cell will not be represented with a separate channel in the output audiogram.
        * applying temporal integration (i.e. smoothing) to the output of the lateral inhibitory network, either by leaky integration (when ``time_constant`` is set to a positive value) or by single time-frame averaging (when ``time_constant`` is set to 0). The output time-frames are contiguous and non-overlapping.

    :param x: The input waveform as a 1-D time-series of audio samples with a maximum sampling frequency of 16kHz. The use of lower
        sampling rates is supported and has to be specified using the ``octave_shift`` parameter. Note, if the length of ``x`` is not
        an integer multiple of the frame length in samples (``frame_length * sampling_rate / 1000``), then zeros will be appended to the end of ``x`` to round up the length to the next integer multiple.
    :type x: numpy.ndarray
    :param octave_shift: At a default sampling rate of 16kHz, the simulated hair-cells span a frequency range of 180-7040 Hz, corresponding 
        to the 64 musical notes F#3-A8 whereby each note is covered by two hair-cells. These :math:`\approx 5.3` octaves will be shifted if the sampling rate
        of ``x`` deviates from its default. In particular, choose sampling rate and ``octave_shift`` in line with the identity
        ``sampling_rate = 16000 * 2 ** octave_shift``. For example, set ``octave_shift`` to ``-1`` for an 8 kHz sampling rate.
    :type octave_shift: int, optional, default=0
    :param frame_length: The length (in milliseconds) of a single output time-frame, i.e. the window over which the filtered
        signal is temporally integrated. Common values: 4, 8, 16, or other powers of two.
    :type frame_length: int, optional, default=4
    :param sigmoid_factor: Controls the non-linear activation function applied to the raw filter outputs.
        Possible values:

        - ``sigmoid_factor > 0``: results in the use of a transistor-like sigmoid of the form :math:`\frac{1}{1 + e^{-x / \mathrm{sigmoid\_factor}}}`, where larger values of ``sigmoid_factor`` result in a flatter slope.
        - ``sigmoid_factor == 0``: results in the use of a step function centred at zero.
        - ``sigmoid_factor == -1``: results in a rectified linear function of the form ``max(0, x)``.
        - other: results in an identity function.
    :type sigmoid_factor: float, optional, default=-2
    :param time_constant: The non-negative time constant (in milliseconds) used for the temporal integration applied to the output of the 
        lateral inhibition network. If ``time_constant > 0``, it results in leaky integration and larger values lead to stronger 
        temporal smoothing of the output. If it is set to ``0``, it results in simple averaging of the sub-samples within a single output time-frame. 
    :type time_constant: int, optional, default=0
    :param verbose: If ``True``, logs a summary of the configuration and inferred sampling rate, plus per-channel progress, at DEBUG
        level through the ``neural_audio.wav2aud`` logger. Nothing is printed unless logging is configured to show DEBUG
        messages, e.g. with ``logging.basicConfig(level=logging.DEBUG)``.
    :type verbose: bool, optional, default=False
    :param filters: If the method should use custom filters instead of the default ones, they can be specified with this dictionary. It is expected to have key:value pairs

        - ``zeros`` : A numpy.ndarray for transfer function zeros with dimensions (maximum number of zeros, 129 filters), ordered by characteristic frequency in ascending fashion.
        - ``poles`` : A numpy.ndarray for transfer function poles with dimensions (maximum number of poles, 129 filters), ordered by characteristic frequency in ascending fashion.
        - ``gain`` : A numpy.ndarray array of length equal to the number of filters, ordered by characteristic frequency in ascending fashion.


        For example, for a filter-bank with 3 filters where the first one has 2 zeros, the second has 5 and the third
        has 3, the value of 'zeros' in the dictionary should correspond to a (5x3) array. Any slots 
        not filled by the zeros of a filter should have NaNs instead. An example matrix for this setup is shown below,
        where ``a``'s, ``b``'s and ``c``'s correspond to the zeros of each filter. The same applies to the 'poles'
        array. The 'gain' array should have length equal to the number of channels (here, 3), and only one dimension.

        .. code-block:: none

            [[a1,  b1, c1 ],
             [a2,  b2, c2 ],
             [NaN, b3, c3 ],
             [NaN, b4, NaN],
             [NaN, b5, NaN]]
             
    :type filters: dict, optional
    :param parallel: If ``True``, the frequency channels are split into independent blocks that are computed simultaneously
        on separate CPU cores. The result is identical to ``parallel=False``. The remaining arguments below only take effect when
        ``parallel=True``.
    :type parallel: bool, optional, keyword-only, default=False
    :param n_jobs: Number of workers to use. ``None`` or ``-1`` uses all CPUs available to the current process.
    :type n_jobs: int, optional, keyword-only, default=None
    :param backend: ``"threads"`` or ``"processes"``. The heavy computations (scipy filtering and numpy array operations) release
        Python's global interpreter lock, so threads already run on separate cores and are usually fastest. ``"processes"``
        starts separate Python processes that each re-import numpy and scipy, which can cost several seconds per call, so
        it only pays off for very long sounds or when threads are not an option. When using
        ``"processes"`` in a script on Windows or macOS, protect the calling code with ``if __name__ == "__main__":``.
    :type backend: str, optional, keyword-only, default="threads"
    :param chunk_size: Number of output channels per subtask. ``None`` splits the channels evenly over ``n_jobs`` workers.
    :type chunk_size: int, optional, keyword-only, default=None

    :returns:
                
        - ``time_points`` (numpy.ndarray) - The ``N`` time points corresponding to the center of each output frame in seconds, where ``N=ceil(len(x) / L)`` and ``L = frame_length * sampling_rate / 1000`` is the frame length in samples.
        - ``frequencies`` (numpy.ndarray) - The ``M=128`` characteristic frequencies of the simulated hair-cells in Hz.
        - ``audiogram`` (numpy.ndarray) - The auditory spectrogram of shape ``[N, M]``, where ``N`` is the number of time-frames and ``M=128`` is the number of frequency channels. This follows the MATLAB NSL toolbox convention (time along the rows, frequency along the columns), so ``audiogram`` can be passed straight into ``aud2cor`` without transposing.
    :rtype: tuple of numpy.ndarray
        
    Example::

        import numpy as np
        from neural_audio.wav2aud import wav2aud

        note = 440 # A4 note in Hz
        fs = 16000 # Sampling rate
        t = np.arange(0, 1, 1/fs) # Input time points
        waveform = np.sin(2 * np.pi * note * t)  # 1 second of audio 
        time_points, frequencies, audiogram  = wav2aud(waveform)

        # Same result, computed on all available CPU cores
        time_points, frequencies, audiogram  = wav2aud(waveform, parallel=True)


    .. warning:: This function has only been tested using the filters of Yang, Wang and Shamma. Providing sounds outside the supported frequency range or using custom filters may lead to unexpected results.
        
    .. rubric:: References

    | Original Author: Powen Ru (powen@isr.umd.edu), NSL, UMD

    .. [1] Yang, X., Wang, K., and Shamma, S. A. (1992). "Auditory
       representations of acoustic signals." *IEEE Transactions on
       Information Theory*, 38(2), 824-839.
       https://doi.org/10.1109/18.119739

    .. [2] Wang, K., and Shamma, S. A. (1994). "Self-normalization and
       noise-robustness in early auditory representations." *IEEE
       Transactions on Speech and Audio Processing*, 2(3), 421-435.
       https://doi.org/10.1109/89.294356
    """

    # --- Input validation ---

    # Ensure x is properly formatted
    assert isinstance(x, np.ndarray) and x.ndim == 1, "Input waveform should be a 1-D numpy array."

    # Ensure octave_shift is an integer
    assert isinstance(octave_shift, int), "The octave_shift parameter should be an integer."

    # Ensure frame_length is a positive integer
    assert isinstance(frame_length, int) and frame_length > 0, "The frame_length parameter should be a positive integer."

    # Ensure sigmoid_factor is a float or an integer
    assert isinstance(sigmoid_factor, (int, float)), "The sigmoid_factor parameter should be a float or integer."

    # Ensure time_constant is a non-negative integer
    assert isinstance(time_constant, int) and time_constant >= 0, "The time_constant parameter should be a non-negative integer."

    # Ensure verbose is a boolean
    assert isinstance(verbose, bool), "The verbose parameter should be a boolean."

    # Ensure parallel is a boolean (the other parallelisation arguments are checked in _wav2aud_parallel)
    assert isinstance(parallel, bool), "The parallel parameter should be a boolean."

    # Second-order sections of all filters, ordered by ascending characteristic frequency
    sos_list = _custom_sos(filters) if filters else _default_sos()
    M = len(sos_list) # Number of filters (one more than the output channels)

    # --- Define constants ---
    # Convenience parameters
    sampling_rate = 16000 * 2**octave_shift
    if verbose:
        logger.debug(f"Converting waveform of length {len(x)/sampling_rate}s samples to cochleagram with \n\t-octave shift {octave_shift}, \n\t-frame length {frame_length} ms, \n\t-sigmoid factor {sigmoid_factor}, \n\t-time constant {time_constant} ms, \n\t-inferred sampling rate {sampling_rate} Hz.")

    L_x = len(x)
    L_frm = int(np.round(frame_length * 2**(4+octave_shift)))     # frame length (samples)

    alph = 0
    if time_constant:
        alph = np.exp(-1 / (time_constant * 2**(4+octave_shift)))  # decay factor for leaky integration

    haircell_tc = 0.5                                  # hair cell time constant (ms)
    beta = np.exp(-1 / (haircell_tc * 2**(4+octave_shift)))   # hair cell membrane decay

    # Zero-pad x to the closest integer multiple of L_frm: N output frames
    N = int(np.ceil(L_x / L_frm))
    x = np.pad(x, (0, int(N * L_frm - L_x)))

    params = dict(sigmoid_factor=sigmoid_factor, beta=beta, alph=alph, L_frm=L_frm, N=N)

    # --- Compute audiogram: N frames x (M-1) channels ---
    if not parallel:
        # Regular wav2aud: process the channels block by block on the current core
        audiogram = np.concatenate([_wav2aud_step(x, sos_list[lo:min(lo + _SERIAL_BLOCK, M-1) + 1], **params)
                                    for lo in range(0, M-1, _SERIAL_BLOCK)], axis=1)
    else:
        audiogram = _wav2aud_parallel(x, sos_list, params, n_jobs=n_jobs, backend=backend, chunk_size=chunk_size, verbose=verbose)

    # Compute frequencies
    frequencies = 440 * 2 ** ((np.arange(M-1) - 31) / 24 + octave_shift) # in Hz, with 440Hz as reference frequency for the 49th channel (A4)
    #frequencies = 1000 * 2 ** (np.linspace(-2.15,2.15,128)) * sampling_rate / 16000
    # Time points for output frames
    time_points = (np.arange(N) + 0.5) * frame_length / 1000  # in seconds

    if verbose:
        logger.debug(f"Finished processing. \n\t-Output audiogram shape (time-frames, frequency channels): {audiogram.shape}, \n\t-time points shape: {time_points.shape}, \n\t-characteristic frequencies: {frequencies[0]} - {frequencies[-1]}Hz.")

    # Outputs
    return time_points, frequencies, audiogram


@lru_cache(maxsize=1)
def _default_sos() -> tuple[np.ndarray, ...]:
    """Backend function, not meant to be called directly. Loads the default filters of Yang, Wang and Shamma once and
    converts them to second-order sections, ordered by ascending characteristic frequency. The result is cached, so later
    calls to :func:`wav2aud` skip the file loading and conversion.
    """
    cochba_file = Path(__file__).parent / 'examples' / 'filters' / 'cochba_filters.npz'
    with np.load(cochba_file, allow_pickle=True) as COCHBA:
        M = int(COCHBA['len'])
        sos_list = tuple(sig.zpk2sos(COCHBA[f'zeros_{ch}'], COCHBA[f'poles_{ch}'], COCHBA[f'gain_{ch}']) for ch in range(M))

    # Note: the cached arrays are shared between calls and must not be modified. They cannot be flagged read-only
    # because scipy's sosfilt rejects read-only coefficient arrays.
    return sos_list


def _custom_sos(filters: dict) -> tuple[np.ndarray, ...]:
    """Backend function, not meant to be called directly. Validates a custom ``filters`` dictionary (see :func:`wav2aud`)
    and converts each filter, i.e. each column of ``zeros`` and ``poles``, to second-order sections.
    """
    assert len(filters) == 3 and {'gain', 'poles', 'zeros'} == set(filters.keys()), "The filters dictionary should have the following keys: 'gain', 'poles', and 'zeros'."
    zeros, poles, gain = np.asarray(filters['zeros']), np.asarray(filters['poles']), np.asarray(filters['gain'])
    assert zeros.ndim == 2 and poles.ndim == 2 and gain.ndim == 1, "The 'zeros' and 'poles' arrays in the filters dictionary should be 2-D and the 'gain' array should be 1-D."
    assert zeros.shape[1] == poles.shape[1] == gain.shape[0], "The length of the 'gain' array in the filters dictionary should be equal to the number of filters (i.e. the second dimension of the 'zeros' and 'poles' arrays)."
    assert gain.shape[0] >= 2, "At least two filters are needed (one more than the number of output channels)."

    sos_list = []
    for ch in range(gain.shape[0]):
        # Each column holds one filter; NaNs pad the unused slots
        z, p = zeros[:, ch], poles[:, ch]
        sos_list.append(sig.zpk2sos(z[~np.isnan(z)], p[~np.isnan(p)], gain[ch]))
    return tuple(sos_list)


def _wav2aud_step(x: np.ndarray,
                  sos_block: tuple[np.ndarray, ...],
                  sigmoid_factor: float,
                  beta: float,
                  alph: float,
                  L_frm: int,
                  N: int) -> np.ndarray:
    """Backend function, not meant to be called directly. Computes a block of adjacent audiogram channels.

    ``sos_block`` holds the second-order sections of ``K + 1`` adjacent filters in ascending frequency order. The last
    filter is only used as the higher neighbour for lateral inhibition, so the block yields ``K`` output channels. Because
    of this one-filter overlap, blocks do not depend on each other and can be computed in parallel.

    :returns: The audiogram block of shape ``[N, K]``.
    """
    # y1 Spatiotemporal displacements along the basilar membrane, one row per filter.
    # sosfilt only accepts one filter per call, so this loop over filters is unavoidable.
    y = np.stack([sig.sosfilt(sos, x) for sos in sos_block])

    # y2 Transduction of ``y1`` into hair cell potentials (or instantaneous auditory nerve firing rate)
    y = mf.sigmoid(y, sigmoid_factor)

    # Hair cell membrane low-pass filter (cutoff <= 4 kHz), applied to all channels at once.
    # Skipped when sigmoid_factor == -2 (linear ionic channel mode).
    if sigmoid_factor != -2:
        y = sig.sosfilt(sig.tf2sos(np.array([1.]), np.array([1, -beta])), y, axis=-1)

    # y3 Lateral inhibitory network: subtract the next higher channel from each channel
    y = y[:-1] - y[1:]

    # y4 Half-wave rectification, modelling the threshold non-linearity in the lateral inhibition network
    np.maximum(y, 0, out=y)

    # Temporal integration
    if alph:
        # Leaky integration
        y = sig.sosfilt(sig.tf2sos(np.array([1.]), np.array([1, -alph])), y, axis=-1)
        out = sig.decimate(y, L_frm, axis=-1)
    elif L_frm == 1:
        out = y
    else:
        # Short-term average over each frame
        out = y.reshape(y.shape[0], N, L_frm).mean(axis=-1)

    return out.T


def _wav2aud_parallel(x: np.ndarray,
                      sos_list: tuple[np.ndarray, ...],
                      params: dict,
                      n_jobs: int | None = None,
                      backend: str = "threads",
                      chunk_size: int | None = None,
                      verbose: bool = False) -> np.ndarray:
    """Backend function, not meant to be called directly. Splits the output channels into contiguous blocks and computes
    them simultaneously with :func:`_wav2aud_step`, using threads or separate processes from :mod:`concurrent.futures`.

    :returns: The audiogram of shape ``[N, len(sos_list) - 1]``.
    """
    # --- Input validation of the parallelisation arguments ---
    n_channels = len(sos_list) - 1
    assert n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool) and (n_jobs == -1 or n_jobs >= 1)), "The n_jobs parameter should be None, -1 or a positive integer."
    assert backend in ("threads", "processes"), "The backend parameter should be either 'threads' or 'processes'."
    assert chunk_size is None or (isinstance(chunk_size, int) and not isinstance(chunk_size, bool) and 1 <= chunk_size <= n_channels), f"The chunk_size parameter should be None or an integer between 1 and {n_channels}."

    # CPUs usable by this process (respects CPU allocations on clusters from Python 3.13 on)
    n_cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    n_jobs = n_cpus if n_jobs in (None, -1) else min(n_jobs, n_cpus)

    # --- Split the task into independent subtasks: blocks of output channels [lo, hi) ---
    if chunk_size is None:
        chunk_size = int(np.ceil(n_channels / n_jobs))
    bounds = [(lo, min(lo + chunk_size, n_channels)) for lo in range(0, n_channels, chunk_size)]
    n_workers = min(n_jobs, len(bounds))

    if verbose:
        logger.debug(f"Computing {len(bounds)} blocks of up to {chunk_size} channels with {n_workers} {backend} worker(s).")

    # Filters lo ... hi, where filter hi is the higher neighbour needed for lateral inhibition of channel hi-1
    blocks = [sos_list[lo:hi + 1] for lo, hi in bounds]

    if n_workers == 1:
        results = [_wav2aud_step(x, block, **params) for block in blocks]
    else:
        if backend == "threads":
            executor = ThreadPoolExecutor(max_workers=n_workers)
        else:
            # "spawn" behaves the same on Windows, macOS and Linux
            executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
        with executor:
            futures = [executor.submit(_wav2aud_step, x, block, **params) for block in blocks]
            results = [future.result() for future in futures]  # in submission order, i.e. ascending frequency

    return np.concatenate(results, axis=1)