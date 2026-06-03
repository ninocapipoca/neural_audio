import numpy as np

def gen_cort(fc, L, STF, PASS=None):
    """
    Generate (bandpass) cortical temporal filter transfer function.

    Parameters
    ----------
    fc : float
        Characteristic frequency (Hz).
    L : int
        Length of the filter (power of 2 preferable).
    STF : float
        Sample rate (frames per second).
    PASS : array-like of length 2, optional
        [idx, K] where idx=1 -> lowpass, 1<idx<K -> bandpass, idx=K -> highpass.
        Defaults to [2, 3] (bandpass).

    Returns
    -------
    H : np.ndarray (complex)
        Filter transfer function of length L.
    """
    if PASS is None:
        PASS = [2, 3]

    t = np.arange(L) / STF * fc # shape (L,)
    h = np.sin(2 * np.pi * t) * t**2 * np.exp(-3.5 * t) * fc

    h = h - np.mean(h)

    H0 = np.fft.fft(h, 2 * L)
    A  = np.angle(H0[:L])
    H  = np.abs(H0[:L])

    maxi = np.argmax(H) # 0-indexed
    H    = H / H[maxi]

    if PASS[0] == 1: # lowpass
        H[:maxi] = 1.0
    elif PASS[0] == PASS[1]: # highpass
        H[maxi + 1:] = 1.0

    H = H * np.exp(1j * A)

    return H

print("testing")
gen_cort(fc, L, STF, PASS=None)