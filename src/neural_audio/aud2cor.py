import numpy as np

def corcplxw(z, fout):
    """Write complex matrix to binary file."""
    data = np.concatenate([np.real(z).ravel(order="F"), np.imag(z).ravel(order="F")])
    fout.write(data.astype(np.float32).tobytes())


def aud2cor(y, para1, rv, sv, fname, DISP=0):
    """
    Cortical rate-scale representation (forward transform).

    Parameters
    ----------
    y     : ndarray (N, M)  — auditory spectrogram
    para1 : list/array      — [paras..., FULLT, FULLX, BP]
    rv    : array           — rate vector in Hz
    sv    : array           — scale vector in cyc/oct
    fname : str             — output filename ('tmp...' = no disk write)
    DISP  : float           — display saturation level (0 = no display)

    Returns
    -------
    cr : ndarray (K2, K1*2, N+2*dN, M+2*dM) — cortical representation
    """

    # --- Parameters ---
    FULLT = para1[4] if len(para1) >= 5 else 0.0
    FULLX = para1[5] if len(para1) >= 6 else FULLT
    BP    = int(para1[6]) if len(para1) >= 7 else 0

    rv = np.asarray(rv).ravel()
    sv = np.asarray(sv).ravel()
    K1 = len(rv)   # number of rate channels
    K2 = len(sv)   # number of scale channels
    N, M = y.shape

    paras = para1[:4]
    STF = 1000.0 / paras[0]           # frames per second
    SRF = 20 if M == 95 else 24       # channels per octave

    # --- FFT padding sizes ---
    N1 = int(2 ** np.ceil(np.log2(N)))
    N2 = N1 * 2
    M1 = int(2 ** np.ceil(np.log2(M)))
    M2 = M1 * 2

    # --- 2D FFT of auditory spectrogram ---
    # First along frequency axis, then along time axis
    Y = np.zeros((N2, M1), dtype=complex)
    for n in range(N):
        R1 = np.fft.fft(y[n, :], M2)
        Y[n, :] = R1[:M1]
    for m in range(M1):
        R1 = np.fft.fft(Y[:N, m], N2)
        Y[:, m] = R1

    # --- Index setup ---
    dM   = int(np.floor(M / 2 * FULLX))
    # frequency margin indices into the M2-length IFFT output
    mdx1 = np.concatenate([
        np.arange(M2 - dM, M2),       # wrap-around (left margin)
        np.arange(0, M + dM)           # main + right margin
    ]).astype(int)

    dN   = int(np.floor(N / 2 * FULLT))
    ndx  = np.arange(0, N + 2 * dN)   # time indices into IFFT output
    ndx1 = ndx                         # same; kept separate to match MATLAB

    # --- Output array ---
    cr = np.zeros((K2, K1 * 2, N + 2 * dN, M + 2 * dM), dtype=complex)

    # --- Open output file ---
    TMP = len(fname) >= 3 and fname[:3] == 'tmp'
    fout = open(fname, 'wb')
    header = np.array(
        list(paras) + [K1, K2] + list(rv) + list(sv) + [N, M, FULLT, FULLX],
        dtype=np.float32
    )
    fout.write(header.tobytes())

    # ------------------------------------------------------------------ #
    # Main loop: rate × direction × scale                                 #
    # ------------------------------------------------------------------ #
    for rdx in range(K1):
        fc_rt = rv[rdx]
        HR = gen_cort(fc_rt, N1, STF, [rdx + 1 + BP, K1 + BP * 2])

        for sgn in [1, -1]:

            if sgn > 0:
                HR = np.concatenate([HR, np.zeros(N1, dtype=complex)])

            else:
                HR = np.concatenate([
                    HR[:1],
                    np.conj(HR[1:N2][::-1])
                ])
                HR[N1] = abs(HR[N1+1])

            # --- First IFFT (along time axis) pulled out of scale loop ---
            z1_freq = np.zeros((N2, M1), dtype=complex)
            for m in range(M1):
                z1_freq[:, m] = HR * Y[:, m]
            z1 = np.fft.ifft(z1_freq, axis=0)   # (N2, M1)
            z1 = z1[ndx1, :]                     # (N+2*dN, M1)

            for sdx in range(K2):
                fc_sc = sv[sdx]
                HS = gen_corf(fc_sc, M1, SRF, [sdx + 1 + BP, K2 + BP * 2])

                # --- Second IFFT (along frequency axis) ---
                z = np.zeros((N + 2 * dN, M + 2 * dM), dtype=complex)
                for n in range(N + 2 * dN):
                    R1 = np.fft.ifft(z1[n, :] * HS.conj(), M2)
                    z[n, :] = R1[mdx1]

                # Store in output array
                col = rdx + (K1 if sgn == 1 else 0)
                cr[sdx, col, :, :] = z

                if not TMP:
                    corcplxw(z, fout)

    fout.close()
    return cr


# ------------------------------------------------------------------ #
# Filter generators — stubs matching the MATLAB originals             #
# ------------------------------------------------------------------ #

def gen_cort(fc, L, STF, PASS=None):
    if PASS is None:
        PASS = [2, 3]

    t = np.arange(L) / STF * fc

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


def gen_corf(fc, L, SRF, KIND=2):

    if np.isscalar(KIND):
        PASS = [2,3]
    else:
        PASS = KIND
        KIND = 2

    R1 = np.arange(L)/L * SRF/2/abs(fc)

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

    return H