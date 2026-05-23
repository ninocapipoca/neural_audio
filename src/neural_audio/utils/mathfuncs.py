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