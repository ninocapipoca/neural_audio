"""aud2cor tests using output from MATLAB equivalent for comparison.

The reference corticograms are reassembled by :mod:`unit_tests.mat_split` from the
sub-100 MB parts committed under each ``cr_halfres_*`` folder. The MATLAB originals are
several hundred megabytes each and are not kept in the repository, so the parts are the
only form these tests read; a case whose parts are missing is dropped.

Reassembly progress is logged to stderr so a long run shows which part it is on::

    python -m unittest unit_tests.aud2cor_split_tests -v
"""

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
from unit_tests.mat_split import load_cortical_reference, reference_exists

test_dir = Path(__file__).parent # unit tests folder
matlab_outputs = test_dir / 'matlab_outputs' / 'aud2cor'

sf = 16000 # sampling frequency

# HALF-RESOLUTION rates and scales
half_rates = 2 ** np.linspace(np.log2(0.5), np.log2(128), 16)   # temporal modulation rates [Hz]
half_scales = 2 ** np.linspace(np.log2(1/5), np.log2(10), 16)   # spectral modulation scales [cyc/oct]


def log(message):
    """Progress line on stderr, where unittest writes its own output."""
    print(f"[aud2cor] {message}", file=sys.stderr, flush=True)


def construct_cases_audio():
    # associates an audio file with the corresponding matlab output for natural sounds
    out = []
    for sound_file in load_sound_file_paths():
        cr_file = matlab_outputs / f"cr_halfres_{sound_file.stem}.mat"
        if reference_exists(cr_file):
            out.append([sound_file, cr_file])

    # do the same for synthetic sounds
    for sound_file in load_sound_file_paths(synthetic=True):
        cr_file = matlab_outputs / 'synthetic' / f"cr_halfres_{sound_file.stem}.mat"
        if reference_exists(cr_file):
            out.append([sound_file, cr_file])

    return out

def name_audio(f, n, p):
    return f"test_audio_split_{str(p[0][0].stem)}"

audio_cases = construct_cases_audio()

log(f"{len(audio_cases)} reference(s) available")
if not audio_cases:
    log("no split references found — build them from the repository root with "
        "`python -m unit_tests.mat_split unit_tests/matlab_outputs/aud2cor`")

class audioSplitTests(unittest.TestCase):
    @parameterized.expand(audio_cases, name_func=name_audio)
    def test_audiofiles(self, sound, mat):
        soundData, sample_rate = soundf.read(sound)

        # get matlab result, reassembled from the split parts
        log(f"{sound.stem}: loading reference")
        started = time.perf_counter()
        matlab_out = load_cortical_reference(mat, verbose=True)
        log(f"{sound.stem}: reference {matlab_out.shape} {matlab_out.dtype.name} "
            f"in {time.perf_counter() - started:.1f} s")

        # get python result
        _, _, spect = wav2aud(soundData)
        py_out = aud2cor(spect, rates=half_rates, scales=half_scales)

        matlab_out = np.transpose(matlab_out, axes=range(matlab_out.ndim)[::-1])

        np.testing.assert_allclose(py_out, matlab_out, atol=1e-2)
