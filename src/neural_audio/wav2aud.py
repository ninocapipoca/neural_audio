import numpy as np
from scipy import signal as sig
from neural_audio.utils import mathfuncs as mf
from pathlib import Path
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def wav2aud(x: np.ndarray, 
            octave_shift: int=0, 
            frame_length: int=4, 
            sigmoid_factor: float=-2, 
            time_constant: int=0, 
            verbose: bool=False, 
            filters: Dict=dict()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    
    """This function computes a biologically inspired spectrogram (also known as audiogram) of an acoustic waveform by simulating the 
    transduction mechanism of the ferret cochlea, following the NSL Matlab toolbox by Yang, Wang and Shamma (1992 and 1994).
    
    A section of the cochlea spanning 128 hair-cells with a frequency selectivity calibrated to the range 180-7040Hz (given
    ``octave_shift=0``) are each simulated by:

        * convolving the signal with a corresponding cell-specific bandpass filter that is implemented as a second-order section infinite impulse response (IIR) function. The frames of the sliding-frame procedure are contiguous and non-overlapping.
        * applying a non-linear activation function to the output of each filter, modeling the hair cell transduction. The nonlinearity can be a sigmoid, step function, rectified linear function, or identity function, depending on the value of ``sigmoid_factor``.
        * applying a low-pass filter to the output of each nonlinearity, modeling the hair cell membrane potential. This step is skipped when ``sigmoid_factor`` is set to -2, which models a linear ionic channel mode without membrane potential dynamics.
        * computing lateral inhibition between adjacent hair-cells by subtracting the output of each filter from the one above it, followed by half-wave rectification. Note, to compute lateral inhibition for the channel of highest frequency, an auxiliary hair-cell with higher frequency selectivity is used. This auxiliary hair-cell will not be represented with a separate channel in the output diagram.
        * applying temporal integration (i.e. smoothing) to the output of the lateral inhibitory network, either by leaky integration (when ``time_constant`` is set to a positive value) or by single time-frame averaging (when ``time_constant`` is set to 0).
    
    :param x: The input waveform as a 1-D time-series of audio samples with a maximum sampling frequency of 16kHz. The use of lower 
        sampling rates is supported and has to be specified using the ``octave_shift`` parameter. Note, if the length of ``x`` is not
        an integer multiple of ``frame_length``, then ``x`` will be pre-padded with zeros to round up the length to that next integer multiple.
    :type x: numpy.ndarray
    :param octave_shift: At a default sampling rate of 16kHz, the simulated hair-cells span a frequency range of 180-7040 Hz, corresponding 
        to the 64 musical notes F#3-A8 whereby each note is covered by two hair-cells. These :math:`\\approx 5.3` octaves will be shifted if the sampling rate
        of ``x`` deviates from its default. In particular, choose sampling rate and ``octave_shift`` in line with the identity
        :math:`sampling_rate = 16K * 2^{octave_shift}`. For example, set ``octave_shift`` to :math:`-1` for an 8 kHz sampling rate.
    :type octave_shift: int, optional, default=0
    :param frame_length: The length (in milliseconds) of a single frame used when convolving ``x`` with the filters. This is equal to the time span 
        of a single output time-frame. Common values: 8, 16, or others powers of two. 
    :type frame_length: int, optional, default=4
    :param sigmoid_factor: Controls the non-linear activation function applied to the raw filter outputs.

        The possible values are:

        * :math:`> 0` -- uses a transistor-like sigmoid of the form :math:`\\frac{1}{1+e^{-x / \\mathrm{sigmoid\\_factor}}}` where larger values of ``sigmoid_factor`` result in a flatter slope.
        * :math:`0` -- uses a step function centred at zero.
        * :math:`-1` -- uses a rectified linear function of the form :math:`\\max(0, x)`.
        * other -- uses the identity function.

    :type sigmoid_factor: float, optional, default=-2
    :param time_constant: The non-negative time constant (in milliseconds) used for the temporal integration applied to the output of the 
        lateral inhibition network. If ``time_constant`` :math:`>0`, it results in leaky integration and larger values lead to stronger 
        temporal smoothing of the output. If it is set to :math:`0`, it results in simple averaging of the sub-samples within a single output time-frame. 
    :type time_constant: int, optional, default=0
    :param verbose: If ``True``, prints summary of configuration, inferred sampling-rate and logs per-channel progress at the DEBUG console.
    :type verbose: bool, optional, default=False
    :param filters: If the method should use custom filters instead of the default ones, they can be specified with this dictionary. It is expected to have key:value pairs
            - 'zeros' : A numpy.ndarray for transfer function zeros with dimensions (maximum number of zeros, 129 filters), ordered by characteristic frequency in ascending fashion.
            - 'poles' : A numpy.ndarray for transfer function zeros with dimensions (maximum number of poles, 129 filters), ordered by characteristic frequency in ascending fashion.
            - 'gain' : A numpy.ndarray array of length equal to the number of filters, ordered by characteristic frequency in ascending fashion. 


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

    :returns: - `frequencies` (numpy.ndarray) - The :math:`M=128` characteristic frequencies of the simulated hair-cells in Hz.
            - `time_points` (numpy.ndarray) - The :math:`N=ceil(len(x) / frame_length)` time points corresponding to the center of each output frame in seconds.
            - `audiogram` (numpy.ndarray) - The auditory spectrogram of shape [:math:`[M,N]`], where :math:`M=128` is the number of frequency channels and :math:`N=ceil(len(x) / frame_length)` is the number of time-frames.
    
    Example::

        import numpy as np
        from neural_audio.wav2aud import wav2aud

        note = 440 # A4 note in Hz
        fs = 16000 # Sampling rate
        t = np.arange(0, 1, 1/fs) # Input time points
        waveform = np.sin(2 * np.pi * note * t)  # 1 second of audio 
        time_points, frequencies, audiogram  = wav2aud(waveform)


    .. note:: This python implementation has been tested on pure sinusoids, object sounds, animal sounds and human speech to produce outputs that are equal to the original NSL toolbox Matlab implementation by Yang, Wang and Shamma (1992 and 1994) up to a numerical precision of 0.01 using the default filters. 

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

    # Ensure filters are in correct format
    if len(filters) == 0 or filters == None:

        # If no filters are specified, load in default filters from file
        cochba_file = Path(__file__).parent / 'examples' / 'filters' / 'cochba_filters.npz'
        COCHBA = np.load(cochba_file, allow_pickle=True)
        M = int(COCHBA['len'])

        z  = COCHBA[f'zeros_{M-1}']
        po = COCHBA[f'poles_{M-1}']
        k  = COCHBA[f'gain_{M-1}']

    # If filters are specified, load them from the dictionary
    else:
        assert len(filters) == 3 and {'gain', 'poles', 'zeros'} == set(filters.keys()), "The filters dictionary should have the following keys: 'gain', 'poles', and 'zeros'."
        assert filters['zeros'].shape == filters['poles'].shape, "The 'zeros' and 'poles' arrays in the filters dictionary should have the same shape."
        assert filters['gain'].shape[0] == filters['zeros'].shape[1], "The length of the 'gain' array in the filters dictionary should be equal to the number of filters (i.e. the second dimension of the 'zeros' and 'poles' arrays)."
        
        M = len(filters['gain']) # Number of filters (one more than the output channels)
        z = filters['zeros'][M-1]
        po = filters['poles'][M-1]
        k = filters['gain'][M-1]
        

    # --- Define constants and allocate output ---
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

    # Allocate output: N frames x (M-1) channels
    N = int(np.ceil(L_x / L_frm))
    x = np.pad(x, (0, int(N * L_frm - L_x))) # zero-pad x to closest integer multiple of L_frm
    x = x.reshape(-1)
    audiogram = np.zeros((N, M-1))


    # --- Highest-frequency channel (channel M-1) ---
    # Process separately: no lateral inhibition at the top of the filterbank.
    if verbose:
        logger.debug(f"processing channel {M-1}")

    sos = sig.zpk2sos(z, po, k)
    y1  = sig.sosfilt(sos, x).squeeze() # y1 Spatiotemporal displacements along the basilar membrane
    y2  = mf.sigmoid(y1, sigmoid_factor) # y2 Transduction of ``y1`` into hair cell potentials (or instantaneous auditory nerve firing rate)

    # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
    # Skipped when sigmoid_factor == -2 (linear ionic channel mode).
    if sigmoid_factor != -2:
        sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
        y2 = sig.sosfilt(sos_y2, y2).squeeze()

    y2_h = y2  # store as reference for lateral inhibition in next channel


    # --- Remaining channels (high -> low frequency) ---
    for ch in range(M-2, 0, -1):
        if verbose:
            logger.debug(f"processing channel {ch}")

        if len(filters) == 0:
            z  = COCHBA[f'zeros_{ch}']
            po = COCHBA[f'poles_{ch}']
            k  = COCHBA[f'gain_{ch}']
        else:
            z = filters['zeros'][ch]
            po = filters['poles'][ch]
            k = filters['gain'][ch]

        sos = sig.zpk2sos(z, po, k)
        y1  = sig.sosfilt(sos, x).squeeze()

        # Transduction: hair cell nonlinearity
        y2 = mf.sigmoid(y1, sigmoid_factor)

        # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
        # Skipped when sigmoid_factor == -2 (linear ionic channel mode).
        if sigmoid_factor != -2:
            sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
            y2 = sig.sosfilt(sos_y2, y2).squeeze()

        # lateral inhibitory network
        y3   = y2 - y2_h # y3 Models lateral inhibitory interactions among LIN neurons
        y2_h = y2
        y4   = np.maximum(y3, 0) # y4 Half-wave rectified ``y3``, modelling threshold non-linearity in the lateral inhibition network

        # Temporal integration
        if alph:
            # Leaky integration
            sos_y5 = sig.tf2sos(np.array([1.]), np.array([1, -alph]))
            y5 = sig.sosfilt(sos_y5, y4).squeeze()
            audiogram[:, ch] = sig.decimate(y5, L_frm)
        else:
            # Short-term average over each frame
            if L_frm == 1:
                audiogram[:, ch] = y4
            else:
                audiogram[:, ch] = np.mean(np.reshape(y4, (N, L_frm)).T, axis=0)

    # Compute frequencies
    frequencies = 440 * 2 ** ((np.arange(M-1) - 31) / 24 + octave_shift) # in Hz, with 440Hz as reference frequency for the 49th channel (A4)
    #frequencies = 1000 * 2 ** (np.linspace(-2.15,2.15,128)) * sampling_rate / 16000
    # Time points for output frames
    time_points = (np.arange(N) + 0.5) * frame_length / 1000  # in seconds

    if verbose:
        logger.debug(f"Finished processing. \n\t-Output audiogram shape: {audiogram.shape}, \n\t-time points shape: {time_points.shape}, \n\t-characteristic frequencies: {frequencies[0]} - {frequencies[0]}Hz.")

    # Outputs
    return time_points, frequencies, audiogram.T