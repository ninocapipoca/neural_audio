import numpy as np
from scipy import signal as sig
from neural_audio.utils import mathfuncs as mf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def wav2aud(x: np.ndarray, 
            frm_len: int=4, 
            time_cst: int=0, 
            sig_fac: float=-2, 
            shift: int=0, 
            verbose: bool=False, 
            cochba_file=None, 
            **kwargs) -> np.ndarray: 
    
    """Compute the auditory spectrogram of an acoustic waveform.

    Converts a raw waveform into an auditory spectrogram based on the modeling
    by Shamma et al and their NSL toobox code originally written in Matlab (see references).
    Relevant for the range of human hearing (20-20000Hz). 

    :param x: Input waveform as a 1-D array of audio samples.
    :type x: numpy.ndarray
    :param frm_len: Frame length. Common values: 8, 16, or
        powers of two. Determines the temporal resolution of the output.
    :type frm_len: int, optional
    :param time_cst: Leaky integration time constant (e.g. 4,
        16, 64). Set to 0 to use short-term frame averaging instead.
    :type time_cst: int, optional
    :param sig_fac: Nonlinear compression factor for the hair cell sigmoid.
        The smaller the value, the greater the compression.
        - `> 0` - transistor-like nonlinearity
        - `0` - full compression (step function)
        - `-1` - half-wave rectifier
        - other - linear function
    :type sig_fac: float, optional
    :param shift: Octave shift relative to 16 kHz. Use ``0`` for 16 kHz,
        ``-1`` for 8 kHz, etc. Sampling frequency = ``16000 * 2^[shft]``
    :type shift: int, optional
    :param verbose: If ``True``, logs per-channel progress at DEBUG level.
    :type verbose: bool, optional
    :param cochba_file: Path to a ``.npz`` file containing the
        cochlear filter bank (poles, zeros, and gains per channel). Defaults to
        the bundled ``cochba_filters.npz``, the filters designed by Shamma et al
        based on their experimental work.
    :type cochba_file: path-like or None, optional
    :returns: Auditory spectrogram of shape ``(N, M-1)``, where ``N`` is the
        number of time frames and ``M-1`` is the number of frequency channels
        (ordered high-to-low frequency).
    :rtype: numpy.ndarray

    .. note::
        The filter bank is applied from the highest to the lowest frequency
        channel. Lateral inhibition is computed by subtracting each channel's
        response from the one above it, followed by half-wave rectification.

        Internal stage variables follow the theoretical work of Yang,
        Shamma, and Wang et al.:

        - ``y1`` — Spatiotemporal displacements along the basilar membrane
        - ``y2`` — Transduction of ``y1`` into hair cell potentials (or
          instantaneous auditory nerve firing rate)
        - ``y3`` — Models lateral inhibitory interactions among LIN neurons
        - ``y4`` — Half-wave rectified ``y3``, modelling threshold
          nonlinearity in the LIN network
        - ``y5`` — Final output of the LIN network

        LIN = lateral inhibitory network.

    Example::

        import numpy as np
        t = np.arange(0, duration + 1/16000, 1/16000)  
        wave = np.sin(2 * np.pi * freq * t[:32000]) # pure sinusoid
        v5 = wav2aud(x, verbose=True)

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
    # TODO - personalize cochba_file such that the user can load in a dictionary
    # containing their own 3 numpy arrays (NUMPY ARRAYS NOT FILES!!!)
    # TODO - add documentation for this
    if not cochba_file:
        # if none specified, set to provided filters
        cochba_file = Path(__file__).parent / 'utils' / 'cochba_filters.npz'

    COCHBA = np.load(cochba_file, allow_pickle=True)
    M = int(COCHBA['len'])

    L_x = len(x)

    L_frm = int(np.round(frm_len * 2**(4+shift)))     # frame length (samples)

    alph = 0
    if time_cst:
        alph = np.exp(-1 / (time_cst * 2**(4+shift)))  # decay factor for leaky integration

    haircell_tc = 0.5                                  # hair cell time constant (ms)
    beta = np.exp(-1 / (haircell_tc * 2**(4+shift)))   # hair cell membrane decay

    # Allocate output: N frames x (M-1) channels
    N = int(np.ceil(L_x / L_frm))
    x = np.pad(x, (0, int(N * L_frm - L_x)))          # zero-pad to multiple of L_frm
    x = x.reshape(-1)
    v5 = np.zeros((N, M-1))

    # --- Highest-frequency channel (channel M-1) ---
    # Process separately: no lateral inhibition at the top of the filterbank.
    z  = COCHBA[f'zeros_{M-1}']
    po = COCHBA[f'poles_{M-1}']
    k  = COCHBA[f'gain_{M-1}']

    if verbose:
        logger.debug(f"processing channel {M-1}")

    sos = sig.zpk2sos(z, po, k)
    y1  = sig.sosfilt(sos, x).squeeze()
    y2  = mf.sigmoid(y1, sig_fac)

    # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
    # Skipped when sig_fac == -2 (linear ionic channel mode).
    if sig_fac != -2:
        sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
        y2 = sig.sosfilt(sos_y2, y2).squeeze()

    y2_h = y2  # store as reference for lateral inhibition in next channel

    # --- Remaining channels (high -> low frequency) ---
    for ch in range(M-2, 0, -1):
        if verbose:
            logger.debug(f"processing channel {ch}")

        z  = COCHBA[f'zeros_{ch}']
        po = COCHBA[f'poles_{ch}']
        k  = COCHBA[f'gain_{ch}']

        sos = sig.zpk2sos(z, po, k)
        y1  = sig.sosfilt(sos, x).squeeze()

        # Transduction: hair cell nonlinearity
        y2 = mf.sigmoid(y1, sig_fac)

        # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
        # Skipped when sig_fac == -2 (linear ionic channel mode).
        if sig_fac != -2:
            sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
            y2 = sig.sosfilt(sos_y2, y2).squeeze()

        # lateral inhibitory network
        y3   = y2 - y2_h 
        y2_h = y2
        y4   = np.maximum(y3, 0)

        # Temporal integration
        if alph:
            # Leaky integration
            sos_y5 = sig.tf2sos(np.array([1.]), np.array([1, -alph]))
            y5 = sig.sosfilt(sos_y5, y4).squeeze()
            v5[:, ch] = sig.decimate(y5, L_frm)
        else:
            # Short-term average over each frame
            if L_frm == 1:
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(np.reshape(y4, (N, L_frm)).T, axis=0)

    return v5