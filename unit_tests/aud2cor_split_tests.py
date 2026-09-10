"""aud2cor tests that read split MATLAB references.

Same comparisons as ``aud2cor_tests.py``, but the reference corticograms are loaded
through :mod:`neural_audio.utils.mat_split`, which reassembles the sub-100 MB parts
committed under each ``cr_halfres_*`` folder.

Which form of the reference gets read is controlled by the ``AUD2COR_REFS`` environment
variable, **not** by what happens to be on disk:

``split`` (default)
    Read only the split parts. If a reference has not been split yet its case is dropped,
    even when the original ``.mat`` is sitting right there — so a green run here really
    does mean the split references are good.
``mat``
    Read only the original ``.mat`` files. Useful for confirming that a failure is in
    ``aud2cor`` rather than in the splitting.
``auto``
    Prefer the parts, fall back to the ``.mat``.

Reassembly progress is logged to stderr so a long run shows which part it is on::

    AUD2COR_REFS=mat python -m unittest unit_tests.aud2cor_split_tests -v
"""

import os
import sys
import time
import unittest
from parameterized import parameterized
from neural_audio.examples.sounds.load import load_sound_file_paths
from pathlib import Path
import numpy as np
import soundfile as soundf
from neural_audio.wav2aud import wav2aud
from neural_audio.aud2cor import aud2cor
from neural_audio.utils.mat_split import (
    SOURCE_CHOICES,
    load_cortical_reference,
    reference_exists,
)

test_dir = Path(__file__).parent # unit tests folder
matlab_outputs = test_dir / 'matlab_outputs' / 'aud2cor'

sf = 16000 # sampling frequency

# HALF-RESOLUTION rates and scales
half_rates = 2 ** np.linspace(np.log2(0.5), np.log2(128), 16)   # temporal modulation rates [Hz]
half_scales = 2 ** np.linspace(np.log2(1/5), np.log2(10), 16)   # spectral modulation scales [cyc/oct]

#: Which reference form to read. Defaults to the split parts; the originals are only used
#: when explicitly asked for, so their presence on disk cannot mask a broken split.
REFERENCE_SOURCE = os.environ.get('AUD2COR_REFS', 'split').strip().lower()
if REFERENCE_SOURCE not in SOURCE_CHOICES:
    raise ValueError(
        f"AUD2COR_REFS must be one of {SOURCE_CHOICES}, got {REFERENCE_SOURCE!r}"
    )


def log(message):
    """Progress line on stderr, where unittest writes its own output."""
    print(f"[aud2cor] {message}", file=sys.stderr, flush=True)


def construct_cases_audio():

    # associates an audio file with the corresponding matlab output for natural sounds
    out = []
    for sound_file in load_sound_file_paths():
        cr_file = matlab_outputs / f"cr_halfres_{sound_file.stem}.mat"
        if reference_exists(cr_file, source=REFERENCE_SOURCE):
            out.append([sound_file, cr_file])

    # do the same for synthetic sounds
    for sound_file in load_sound_file_paths(synthetic=True):
        cr_file = matlab_outputs / 'synthetic' / f"cr_halfres_{sound_file.stem}.mat"
        if reference_exists(cr_file, source=REFERENCE_SOURCE):
            out.append([sound_file, cr_file])

    return out

def name_audio(f, n, p):
    return f"test_audio_split_{str(p[0][0].stem)}"

audio_cases = construct_cases_audio()

log(f"AUD2COR_REFS={REFERENCE_SOURCE!r}: {len(audio_cases)} reference(s) available")
if not audio_cases and REFERENCE_SOURCE == 'split':
    log("no split references found — build them with "
        "`python -m neural_audio.utils.mat_split unit_tests/matlab_outputs/aud2cor`")

class audioSplitTests(unittest.TestCase):
    @parameterized.expand(audio_cases, name_func=name_audio)
    def test_audiofiles(self, sound, mat):
        soundData, sample_rate = soundf.read(sound)

        # get matlab result, reassembled from the split parts unless AUD2COR_REFS says
        # otherwise; progress is logged because a reassembly moves hundreds of MB
        log(f"{sound.stem}: loading reference ({REFERENCE_SOURCE})")
        started = time.perf_counter()
        matlab_out = load_cortical_reference(
            mat, source=REFERENCE_SOURCE, progress=log,
        )
        log(f"{sound.stem}: reference {matlab_out.shape} {matlab_out.dtype.name} "
            f"in {time.perf_counter() - started:.1f} s")

        # get python result
        _, _, spect = wav2aud(soundData)
        py_out = aud2cor(spect, rates=half_rates, scales=half_scales)

        matlab_out = np.transpose(matlab_out, axes=range(matlab_out.ndim)[::-1])

        np.testing.assert_allclose(py_out, matlab_out, atol=1e-2)
