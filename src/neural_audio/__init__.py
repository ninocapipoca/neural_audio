"""Bio-inspired sound processing.

An implementation of the auditory model of Shamma et al., originally written in
MATLAB by Powen Ru and colleagues (Neural Systems Laboratory, University of
Maryland).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from neural_audio.aud2cor import aud2cor, gen_corf, gen_cort
from neural_audio.wav2aud import wav2aud

try:
    __version__ = _version("neural_audio")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.1.0"

__all__ = ["wav2aud", "aud2cor", "gen_cort", "gen_corf", "__version__"]
