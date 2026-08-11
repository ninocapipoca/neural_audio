import numpy as np
import warnings

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
               seed: int = 0) -> np.ndarray:
    r"""
    Generates a sinusoidal ripple stimulus following the description in Chi et al. (2005), caption of Fig. 1(b):

    :math:`S(t, x) = 1 + \sin (2\pi(wt + \Omega x))`

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
    :type duration: float, optional, default=2.0

    :param sf: Sampling frequency, in Hz.
    :type sf: int, optional, default=16000

    :param f_min: Lowest carrier (sinusoid) frequency, in Hz.
    :type f_min: float, optional, default=180

    :param f_max: Highest carrier (sinusoid) frequency, in Hz.
    :type f_max: float, optional, default=7040

    :param seed: Seed for the random carrier phases. A fixed value keeps the output
        reproducible; pass a different value for an independent realisation.
    :type seed: int, optional, default=0

    :returns: 1D signal of shape (duration*sf,).
    :rtype: numpy.ndarray

    .. note:: Two design choices work together to suppress interference artefacts in the audiogram.
        First, each carrier is given a random starting phase (rather than all starting in phase at
        :math:`t=0`), which prevents coherent onset transients from producing a deterministic,
        frequency-swept interference ("fingerprint") pattern near the onset. Second, the carrier
        grid is deliberately dense (1024 log-spaced carriers): adjacent carriers that fall within
        the same cochlear filter's bandwidth beat at their frequency difference regardless of phase,
        so many carriers per filter are needed for those pair-beats to add incoherently and integrate
        out. Together these approximate the broadband-noise carrier described by Chi et al. (2005)
        and leave only the intended diagonal ripple.

    .. rubric:: References

    .. [1] Chi, Taishih & Ru, Powen & Shamma, Shihab (2005),
        "Multiresolution spectrotemporal analysis of complex sounds" ,
        The Journal of the Acoustical Society of America, 118, 887-906,
        10.1121/1.1945807
    """
    num_channels = 1024  # dense enough that pair-beats within a cochlear filter cancel (see note)
    t = np.arange(0, duration, 1 / sf)
    freqs = np.geomspace(f_min, f_max, num_channels)
    rel_pos = np.log2(freqs / f_min)  # position in octaves, 0 at f_min

    # random per-carrier phases so carriers do not all cohere at t=0 (see note)
    rng = np.random.default_rng(seed)
    carrier_phases = rng.uniform(0, 2 * np.pi, size=num_channels)

    # Pre-compute the shared 2*pi*t factor and the time part of the ripple envelope
    # argument (independent of the carrier index), so the loop only does per-carrier work.
    tau = 2 * np.pi * t
    env_time = rate * tau

    signal = np.zeros_like(t)
    for f, pos, cp in zip(freqs, rel_pos, carrier_phases):
        envelope = 1 + np.sin(env_time + 2 * np.pi * scale * pos)
        signal += envelope * np.sin(f * tau + cp)

    return signal

def gen_temporal_modulations_rate(rate: float, duration: float = 2, sf: int = 16000,
                                  carrier_freq: float = 1000, depth: float = 1.0, ramp: bool = False, 
                                  ramp_frac: float = 0.05, clip=True) -> np.ndarray:
    """
    Generates a signal containing only temporal modulations at a specified
    modulation rate.

    A pure-tone carrier is amplitude-modulated by a sinusoidal envelope at
    ``rate`` Hz. Because the carrier is a single frequency, the modulation
    appears in one horizontal band of the audiogram (the channels tuned near
    ``carrier_freq``). The sinusoidal envelope produces smooth temporal
    modulation without the sharp broadband onsets of an impulse train.

    :param rate: Temporal modulation rate in Hz.
    :type rate: float

    :param duration: Signal duration in seconds.
    :type duration: float, optional

    :param sf: Sampling frequency in Hz.
    :type sf: int, optional

    :param carrier_freq: Carrier tone frequency in Hz. Should fall within the
        range the cochlear filterbank covers (roughly 180-7040 Hz at the
        default sampling rate). Default is 1kHz.
    :type carrier_freq: float, optional

    :param depth: Modulation depth in [0, 1]. At 1.0 the envelope reaches zero
        at its troughs (100% modulation); at 0.0 the carrier is unmodulated.
    :type depth: float, optional

    :param ramp: Whether to use a Hann window to reduce edge artefacts. Note that
        this adds some slow spectral modulation, so this is `False` by default.
    :type ramp: bool, optional

    :param ramp_frac: Fraction of total length to taper (use as a ramp) at each end
        of the original signal. A small value is recommended (0.05 by default), and the maximum possible value is
        0.5, which applies the Hann window to the entire signal.
    :type ramp_frac: float, optional

    :param clip: Whether to clip the carrier frequency if it falls outside the approximate default filterbank range. True by
        default, but can be toggled to allow compatibility with other filterbanks.
    :type clip: bool, optional

    :returns: 1-D signal of length ``int(duration * sf)``.
    :rtype: numpy.ndarray
    """

    if not 0 <= ramp_frac <= 0.5:
        raise ValueError("ramp_frac parameter must be between 0 and 0.5")

    if not 0 <= depth <= 1:
        raise ValueError("depth parameter must be between 0 and 1")

    if not 180 <= carrier_freq <= 7040:
        warnings.warn(f"carrier_freq={carrier_freq} Hz is outside or close to the edges of the default filterbank range (approx 180-7040 Hz)")
        if clip:
            carrier_freq = np.clip(carrier_freq, 180, 7040)
            warnings.warn(f"carrier_freq= clipped to {carrier_freq}")

    if carrier_freq < 8 * rate: 
        # a rough baseline to avoid artefacts
        warnings.warn(f"carrier_freq={carrier_freq} Hz is not much greater than rate={rate} Hz; envelope may not be well-defined (beating). Try {carrier_freq*8}Hz")

    t = np.arange(0, duration, 1 / sf)

    # pure-tone carrier at a single frequency
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    # sinusoidal amplitude envelope at the modulation rate.
    # cosine starts at a peak; shifted so the envelope stays in [1-depth, 1].
    envelope = 1 - depth * (1 - np.cos(2 * np.pi * rate * t)) / 2

    if ramp:
        ramp_len = int(ramp_frac * len(t))
        ramp_func = np.hanning(2 * ramp_len)
        window = np.ones(len(t))
        window[:ramp_len] = ramp_func[:ramp_len]
        window[-ramp_len:] = ramp_func[ramp_len:]
        envelope = envelope * window

    signal = envelope * carrier

    return signal, envelope

def gen_temporal_bursts_rate(rate: float, duration: float = 2, sf: int = 16000) -> np.ndarray:
    """
    Generates a signal containing bursts at a specified
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

def gen_spectral_modulations_scale(scale: float, f_min: int = 180, f_max: int = 7040,
                                   duration: int = 2, sf: int = 16000) -> np.ndarray:
    """
    Generates a signal with only spectral modulations at a given spectral modulation scale.

    This is a thin wrapper around :func:`gen_spectral_modulations`: the ``scale`` (in
    cycles/octave) sets how many log-spaced sinusoidal carriers are placed per octave across
    the ``[f_min, f_max]`` range, namely ``num_sinusoids = round(scale * n_octaves) + 1``.
    A larger ``scale`` therefore packs the carriers more densely, giving more closely spaced
    spectral stripes (faster variation across frequency). As with ``gen_spectral_modulations``,
    the signal is constant in time.

    :param scale: Spectral modulation scale in cycles/octave. Sets the carrier density
        (carriers per octave).
    :type scale: float

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
        For non-default paramters, you may need to adjust ``f_min`` and ``f_max`` accordingly.
    """
    n_octaves = np.log2(f_max / f_min)
    num_sinusoids = max(2, round(scale * n_octaves) + 1)

    return gen_spectral_modulations(num_sinusoids=num_sinusoids, f_min=f_min,
                                    f_max=f_max, duration=duration, sf=sf)
