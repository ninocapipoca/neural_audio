import numpy as np

def corcplxw(z, fout):
    """Write complex matrix to binary file."""
    data = np.concatenate([np.real(z).ravel(order="F"), np.imag(z).ravel(order="F")])
    fout.write(data.astype(np.float32).tobytes())

def aud2cor(y: np.ndarray, 
            octave_shift: int=0, 
            frame_length: int=4, 
            sigmoid_factor: float=-2, 
            time_constant: int=0,
            tp_margin: float=0, # fullness of temporal margin
            sp_margin: float=0, # fullness of spectral margin
            bandpass: float=1,
            spfilt_ripples: np.ndarray=None, #rv equivalent
            tpfilt_cf: np.ndarray=None, #sv equivalent
            fname: str=''): 
    """
    Cortical rate-scale representation (forward transform).

    Parameters
    ----------
    :param y: The result of wav2aud; the auditory spectrogram of shape [:math:`[N,M]`], where :math:`M=128` is the number of frequency channels and :math:`N=ceil(len(x) / frame_length)` is the number of time-frames.
    :type y: numpy.ndarray

    :param tp_margin: Fullness of the temporal margin; any real value in [0,1].
    :type tp_margin: float


    Returns
    -------
    cr : ndarray (num_scales, num_rates*2, N+2*dN, M+2*dM) — cortical representation
    """

    # --- Parameters ---
    #para1 = [paras tp_margin sp_margin bandpass]
    #paras = [frmlen, tc, fac, shft]
    # complete params = [frmlen, tc, fac, shft, tp_margin, sp_margin, bandpass]

    # TODO Data types check

    # Check margins are within allowed range
    if not (0 <= tp_margin <= 1):
        raise ValueError(f"tp_margin must be in [0, 1], got {tp_margin}")
    if not (0 <= sp_margin <= 1):
        raise ValueError(f"sp_margin must be in [0, 1], got {sp_margin}")
    
    # set default spfilt and tpfilt params
    if spfilt_ripples is None:
        spfilt_ripples = 2 ** np.linspace(np.log2(0.5), np.log2(128), 32)
    if tpfilt_cf is None:
        tpfilt_cf = 2 ** np.linspace(np.log2(1/5), np.log2(10), 32)

    tpfilt_cf = np.asarray(tpfilt_cf).ravel()
    spfilt_ripples = np.asarray(spfilt_ripples).ravel()
    num_rates = len(tpfilt_cf) # number of rate channels K1
    num_scales = len(spfilt_ripples) # number of scale channels K2
    N, M = y.shape # dimensions of audiogram; shape (timepoints, num channels)

    fps = 1000.0 / frame_length # frames per second (fps)
    ch_per_oct = 20 if M == 95 else 24 # channels per octave (ch_per_oct)

    # --- FFT padding sizes ---
    N_pad = int(2 ** np.ceil(np.log2(N)))
    #N2 = N_pad * 2
    M_pad = int(2 ** np.ceil(np.log2(M)))
    #M2 = M_pad * 2

    # --- 2D FFT of auditory spectrogram ---
    # First along frequency axis, then along time axis
    Y = np.zeros((N_pad*2, M_pad), dtype=complex)
    for n in range(N):
        R1 = np.fft.fft(y[n, :], 2*M_pad)
        Y[n, :] = R1[:M_pad]
    for m in range(M_pad): # !! fft on output of previous (Y)
        # NOTE Allows you to capture interaction
        R1 = np.fft.fft(Y[:N, m], 2*N_pad)
        Y[:, m] = R1

    # --- Index setup ---
    dM   = int(np.floor(M / 2 * sp_margin))

    # frequency margin indices into the (2*M_pad)-length IFFT output
    mdx1 = np.concatenate([
        np.arange((2*M_pad) - dM, (2*M_pad)), # wrap-around (left margin)
        np.arange(0, M + dM) # main + right margin
    ]).astype(int)

    dN   = int(np.floor(N / 2 * tp_margin))
    ndx  = np.arange(0, N + 2 * dN)   # time indices into IFFT output
    ndx1 = ndx                         # same; kept separate to match MATLAB

    # --- Output array ---
    cr = np.zeros((num_scales, num_rates * 2, N + 2 * dN, M + 2 * dM), dtype=complex)

    # NOTE - Consider separating file handling
    # --- Open output file ---
    write_file = len(fname) > 0
    if write_file:
        fout = open(fname, 'wb')
        header = np.array(
            [frame_length, time_constant, sigmoid_factor, octave_shift] + [num_rates, num_scales] + list(tpfilt_cf) + list(spfilt_ripples) + [N, M, tp_margin, sp_margin],
            dtype=np.float32
        )
        fout.write(header.tobytes())

    # ------------------------------------------------------------------ #
    # Main loop: rate × direction × scale                                #
    # ------------------------------------------------------------------ #
    for rdx in range(num_rates):
        fc_rt = tpfilt_cf[rdx]
        HR = gen_cort(fc_rt, N_pad, fps, [rdx + 1 + bandpass, num_rates + bandpass * 2])

        for sgn in [1, -1]:

            if sgn > 0:
                HR = np.concatenate([HR, np.zeros(N_pad, dtype=complex)])

            else:
                HR = np.concatenate([HR[:1], np.conj(HR[1:2*N_pad][::-1])])
                HR[N_pad] = abs(HR[N_pad+1])

            # --- First IFFT (along time axis) pulled out of scale loop ---
            z1_freq = np.zeros((2*N_pad, M_pad), dtype=complex)
            for m in range(M_pad):
                z1_freq[:, m] = HR * Y[:, m]
            z1 = np.fft.ifft(z1_freq, axis=0)   # (2*N_pad, M_pad) trying to match original freq and time to s.t. modulation patterns found
            z1 = z1[ndx1, :]                     # (N+2*dN, M_pad)

            for sdx in range(num_scales):
                fc_sc = spfilt_ripples[sdx]
                HS = gen_corf(fc_sc, M_pad, ch_per_oct, [sdx + 1 + bandpass, num_scales + bandpass * 2])

                # --- Second IFFT (along frequency axis) ---
                z = np.zeros((N + 2 * dN, M + 2 * dM), dtype=complex)
                for n in range(N + 2 * dN):
                    R1 = np.fft.ifft(z1[n, :] * HS.conj(), (2*M_pad))
                    z[n, :] = R1[mdx1]

                # Store in output array
                col = rdx + (num_rates if sgn == 1 else 0)
                cr[sdx, col, :, :] = z

                if write_file:
                    corcplxw(z, fout)
    if write_file:
        fout.close()
    return cr


# ------------------------------------------------------------------ #
# Filter generators — stubs matching the MATLAB originals             #
# ------------------------------------------------------------------ #

def gen_cort(fc, L, fps, PASS=None):
    if PASS is None:
        PASS = [2, 3]

    t = np.arange(L) / fps * fc
    h = np.sin(2*np.pi*t) * t**2 * np.exp(-3.5*t) * fc

    h = h - np.mean(h)

    H0 = np.fft.fft(h, 2*L)

    A = np.angle(H0[:L])
    H = np.abs(H0[:L])

    maxi = np.argmax(H)
    H /= H[maxi]

    if PASS[0] == 1:
        H[:maxi] = 1
    elif PASS[0] == PASS[1]:
        H[maxi+1:] = 1

    return H * np.exp(1j*A)


def gen_corf(fc, L, ch_per_oct, KIND=2):

    if np.isscalar(KIND):
        PASS = [2,3]
    else:
        PASS = KIND
        KIND = 2

    R1 = np.arange(L)/L * ch_per_oct/2/abs(fc)

    if KIND == 1:
        C1 = 1/(2*.3*.3)
        H = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)
    else:
        R1 = R1**2
        H = R1*np.exp(1-R1)

    if PASS[0] == 1:
        maxi = np.argmax(H)
        s = np.sum(H)
        H[:maxi] = 1
        H = H/np.sum(H)*s

    elif PASS[0] == PASS[1]:
        maxi = np.argmax(H)
        s = np.sum(H)
        H[maxi+1:] = 1
        H = H/np.sum(H)*s

    # TODO - return rates and scales as well (compute explicitly)
    return H