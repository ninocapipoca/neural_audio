import unittest
from parameterized import parameterized
from neural_audio.examples.sounds.load import load_sound_file_paths
from pathlib import Path
import numpy as np
import h5py
import soundfile as soundf
from neural_audio.wav2aud import wav2aud
from neural_audio.aud2cor import aud2cor

test_dir = Path(__file__).parent # unit tests folder
matlab_outputs = test_dir / 'matlab_outputs' / 'aud2cor'

sf = 16000 # sampling frequency

# HALF-RESOLUTION rates and scales
half_rates = 2 ** np.linspace(np.log2(0.5), np.log2(128), 16)   # temporal modulation rates [Hz]
half_scales = 2 ** np.linspace(np.log2(1/5), np.log2(10), 16)   # spectral modulation scales [cyc/oct]

def construct_cases_audio():

    # associates an audio file with the corresponding matlab output for natural sounds
    out = []
    for sound_file in load_sound_file_paths():
        cr_file = matlab_outputs / f"cr_halfres_{sound_file.stem}.mat"
        if cr_file.exists():
            out.append([sound_file, cr_file])

    # do the same for synthetic sounds
    for sound_file in load_sound_file_paths(synthetic=True):
        cr_file = matlab_outputs / 'synthetic' / f"cr_halfres_{sound_file.stem}.mat"
        if cr_file.exists():
            out.append([sound_file, cr_file])
    
    return out

def name_audio(f, n, p):
    return f"test_audio_{str(p[0][0].stem)}"

audio_cases = construct_cases_audio()

class audioTests(unittest.TestCase):
    @parameterized.expand(audio_cases, name_func=name_audio)
    def test_audiofiles(self, sound, mat):
        soundData, sample_rate = soundf.read(sound)

        # get matlab result
        bin_file = h5py.File(mat, 'r')
        matlab_out = bin_file['cr'][:]
        matlab_out = matlab_out['real'] + 1j * matlab_out['imag']

        # get python result
        _, _, spect = wav2aud(soundData)
        py_out = aud2cor(spect, rates=half_rates, scales=half_scales)

        matlab_out = np.transpose(matlab_out, axes=range(matlab_out.ndim)[::-1])
        
        np.testing.assert_allclose(py_out, matlab_out, atol=1e-2)