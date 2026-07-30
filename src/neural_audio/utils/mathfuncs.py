import numpy as np

def sigmoid(y: np.ndarray, fac:int) -> np.ndarray:
    """
    Nonlinear sigmoid function for cochlear hair cell modelling.

    A monotonic increasing function that simulates hair cell nonlinearity.
    Behaviour is controlled by the ``fac`` parameter, which selects between
    several operating modes ranging from a smooth transistor-like response
    to a hard limiter or half-wave rectifier.

    :param y: Input signal array.
    :type y: numpy.ndarray
    :param fac: Nonlinear factor controlling the operating mode:

        - ``fac > 0`` — transistor-like sigmoid (smooth nonlinearity)
        - ``fac = 0`` — hard limiter (Heaviside step function)
        - ``fac = -1`` — half-wave rectifier (clips negative values to zero)
        - otherwise — linear passthrough (no operation)

    :type fac: float
    :returns: Transformed signal, same shape as ``y``.
    :rtype: numpy.ndarray

    .. seealso::
        :func:`wav2aud`

    .. rubric:: References

    | Original Author: Powen Ru (powen@isr.umd.edu), NSL, UMD

    """

    if fac > 0:
        return 1/(1 + np.exp(-y / fac))
    elif fac == 0:
        return (y > 0).astype('float')
    elif fac == -1:
        return np.maximum(y, 0)
    else:
        return y


def gen_temporal_modulations(num_bursts: int=None, duration: int=2, sf: int=16000) -> np.ndarray:
    """
    Generates a signal with only temporal modulations - every frequency is present but only at infinitesimally 
    small, evenly spaced bursts. 
    
    Useful to build test cases or for early exploration of the toolbox.

    :param num_bursts: The number of bursts in the signal. If none specified, is set to ``np.ceil(duration/5)``
    :type num_bursts: int, optional, default=None

    :param duration: Duration of the signal, in seconds
    :type duration: int, optional, default=2

    :param sf: Sampling frequency in Hz, used in the calculation of the number of time points (``duration * sf``) 
        and hence defines the signal's temporal resolution
    :type sf: int, optional, default=16000

    :returns: 1D signal of shape (duration*sf,); for example, if your sampling frequency is 16kHz and you have a duration of
    2 seconds, your output signal will have length 32kHz.
    :rtype: numpy.ndarray
    """
    t = np.arange(0, duration, 1/sf) 

    if num_bursts is None:
        num_bursts = int(np.ceil(duration/5))

    # create short evenly spaced bursts
    v_stripes = np.zeros_like(t)
    burst_idxs = np.linspace(0, len(t) - 1, num_bursts, dtype=int)
    v_stripes[burst_idxs] = 1.0

    return v_stripes

def gen_spectral_modulations(num_sinusoids: int=6, 
                             f_min: int=180,
                             f_max: int=7040,
                             duration: int=2, 
                             sf: int=16000) -> np.ndarray:
    """
    Generates a signal with only spectral modulations - different frequencies are present but they are constant in time.
    Useful to build test cases or for early exploration of the toolbox.

    :param num_sinusoids: The number of different frequencies present in the signal. These will be linearly spaced on the 
        log scale.
    :type num_sinusoids: int, optional, default=6

    :param f_min: The minimum frequency present in the signal, in Hz
    :type f_min: int, optional, default=180

    :param f_max: The maximum frequency present in the signal, in Hz
    :type f_max: int, optional, default=7040

    :param duration: Duration of the signal, in seconds
    :type duration: int, optional, default=2

    :param sf: Sampling frequency in Hz, used in the calculation of the number of time points (``duration * sf``) 
        and hence defines the signal's temporal resolution
    :type sf: int, optional, default=16000

    :returns: 1D signal of shape (duration*sf,); for example, if your sampling frequency is 16kHz and you have a duration of
    2 seconds, your output signal will have length 32kHz.
    :rtype: numpy.ndarray

    .. warning:: If you plan on using this signal with ``wav2aud`` make sure that the maximum and minimum frequencies are
        within the calibrated range. For the default ``wav2aud`` parameters (octave shift 0, sampling rate 16kHz) the range is 180-7040Hz. 
        For non-default paramters, you may need to adjust ``f_min`` and ``f_max`` accordingly. See ``wav2aud`` documentation or 
        the corresponding tutorial for more information on these adjustments.
    """
    t = np.arange(0, duration, 1/sf) 

    h_stripes = 0
    for freq in np.geomspace(f_min, f_max, num_sinusoids):
        h_stripes += np.sin(2 * np.pi * freq * t)

    return h_stripes

def gen_ripple(rate: float, 
               scale: float, 
               duration: float = 2.0, 
               sf: int = 16000,
               f_min: float = 180, 
               f_max: float = 7040, 
               num_channels: int = 128,
               mod_depth: float = 0.9, 
               phase: float = 0.0) -> np.ndarray:
    r"""
    Generates a sinusoidal ripple stimulus following the description in Chi et al. (2005), caption of Fig. 1(b):

    :math:`S(t, x) = 1 + A \sin (2\pi(wt + \Omega x) + \Phi)`

    where x is position along the log-frequency axis in octaves relative to `f_min`,
    w (rate) is the ripple velocity in Hz, and Omega (scale) is the ripple density
    in cycles/octave.

    A ripple stimulus (the signal) is made up of many pure sinusoids, log-spaced in frequency between `f_min` and `f_max`, 
    which are each amplitude-modulated by S(t, x) evaluated at their own octave position x, then summed.
    Here, S(t,x) is referred to as the ripple function, which is not the same as the signal
    itself; it only describes how the signal changes across time and frequency.

    :param rate: Ripple velocity (w), in Hz. Controls how quickly the stimulus modulations happen in time. 
        Sign controls sweep direction (up/down).
    :type rate: float

    :param scale: Ripple density (Omega), in cycles/octave. Controls how frequency modulations in the stimulus occur.
    :type scale: float

    :param duration: Duration of the signal, in seconds.
    :type duration: float

    :param sf: Sampling frequency, in Hz.
    :type sf: int, optional, default=16000

    :param f_min: Lowest carrier (sinusoid) frequency, in Hz.
    :type f_min: float, optional, default=180

    :param f_max: Highest carrier (sinusoid) frequency, in Hz.
    :type f_max: float, optional, default=7040

    :param num_channels: Number of log-spaced carrier sinusoids spanning f_min-f_max.
    :type num_channels: int, optional, default=128

    :param mod_depth: Modulation depth (A). Should be <= 1 to avoid negative envelope values. 
        Controls the amplitude of the ripple curve. Controls how extreme the changes in the 
        stimulus are (i.e, the contrast).
    :type mod_depth: float, optional, default=0.9

    :param phase: Ripple phase (Phi), in radians. Shifts ripple curve, changing where the 
        peaks and troughs sit relative to the frequency channels.
    :type phase: float, optional, default=0

    :returns: 1D signal of shape (duration*sf,).
    :rtype: numpy.ndarray

    .. rubric:: References

    .. [1] Chi, Taishih & Ru, Powen & Shamma, Shihab (2005),
        "Multiresolution spectrotemporal analysis of complex sounds" ,
        The Journal of the Acoustical Society of America, 118, 887-906, 
        10.1121/1.1945807
    """
    t = np.arange(0, duration, 1 / sf)

    # log-spaced carrier frequencies and their octave positions
    freqs = np.geomspace(f_min, f_max, num_channels)
    rel_pos = np.log2(freqs / f_min) # relative position in octaves, 0 at f_min

    signal = np.zeros_like(t)
    for f, pos in zip(freqs, rel_pos):
        envelope = 1 + mod_depth * np.sin(2 * np.pi * (rate * t + scale * pos) + phase)
        signal += envelope * np.sin(2 * np.pi * f * t)

    #signal /= num_channels
    return signal

def gen_temporal_modulations_rate(rate: float, duration: float = 2, sf: int = 16000) -> np.ndarray:
    """
    Generates a signal containing only temporal modulations at a specified
    modulation rate.

    The signal consists of evenly spaced impulses, with ``rate`` bursts per
    second. Every acoustic frequency is therefore present, while the temporal
    modulation is controlled solely by the burst spacing.

    :param rate: Temporal modulation rate in Hz (bursts per second).
    :type rate: float

    :param duration: Signal duration in seconds.
    :type duration: float, optional

    :param sf: Sampling frequency in Hz.
    :type sf: int, optional

    :returns: 1-D signal of length ``duration * sf``.
    :rtype: numpy.ndarray
    """
    t = np.arange(0, duration, 1 / sf)

    num_bursts = max(1, int(np.round(rate * duration)))

    signal = np.zeros_like(t)
    burst_idxs = np.linspace(0, len(t) - 1, num_bursts, dtype=int)
    signal[burst_idxs] = 1.0

    return signal

def gen_spectral_modulations_scale(scale: float, f_min: float = 180, f_max: float = 7040,
                                   duration: float = 2, sf: int = 16000, num_channels: int = 128,
                                   seed: int = 0) -> np.ndarray:
    """
    Generates a signal containing only spectral modulations at a specified
    spectral modulation scale.

    The signal is formed by summing log-spaced sinusoidal carriers whose amplitudes
    follow a sinusoidal envelope across the log-frequency axis. Each carrier is given
    a random starting phase (see note below) so that the resulting audiogram shows
    the intended horizontal spectral stripes without interference artefacts.

    :param scale: Spectral modulation scale in cycles/octave.
    :type scale: float

    :param f_min: Lowest carrier frequency (Hz).
    :type f_min: float

    :param f_max: Highest carrier frequency (Hz).
    :type f_max: float

    :param duration: Signal duration in seconds.
    :type duration: float

    :param sf: Sampling frequency in Hz.
    :type sf: int

    :param num_channels: Number of log-spaced carrier frequencies.
    :type num_channels: int

    :param seed: Seed for the random carrier phases. A fixed value keeps the output
        reproducible; pass a different value for an independent realisation.
    :type seed: int, optional, default=0

    :returns: 1-D signal of length ``duration * sf``.
    :rtype: numpy.ndarray

    .. note:: The carriers are given random starting phases rather than all starting
        in phase at :math:`t=0`. Densely log-spaced carriers fall within each cochlear
        filter's bandwidth and beat against one another; when they share a common phase
        origin this beating is coherent and produces a deterministic, frequency-swept
        interference ("fingerprint") pattern in the audiogram. Randomising the phases
        decorrelates the beating (approximating the broadband-noise carrier described by
        Chi et al., 2005) and leaves only the intended horizontal spectral stripes.
    """
    t = np.arange(0, duration, 1 / sf)

    freqs = np.geomspace(f_min, f_max, num_channels)

    # position in octaves relative to f_min
    octave_pos = np.log2(freqs / f_min)

    # random per-carrier phases so carriers do not all cohere at t=0 (see note)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=num_channels)

    signal = np.zeros_like(t)

    for f, x, phi in zip(freqs, octave_pos, phases):
        amplitude = np.sin(2 * np.pi * scale * x)
        signal += amplitude * np.sin(2 * np.pi * f * t + phi)

    return signal
