import unittest
from parameterized import parameterized
from neural_audio.examples.sounds.load import load_sound_file_paths2
from pathlib import Path
import numpy as np
import h5py
import soundfile as soundf
from neural_audio.wav2aud import wav2aud
from neural_audio.aud2cor import aud2cor

test_dir = Path(__file__).parent # unit tests folder
matlab_outputs = test_dir / 'matlab_outputs' / 'aud2cor'

sf = 16000 # sampling frequency

def construct_cases_audio():

    # associates an audio file with the corresponding matlab output for natural sounds
    out = []
    for sound_file in load_sound_file_paths2():
        cr_file = matlab_outputs / f"cr_{sound_file.stem}.mat"
        if cr_file.exists():
            out.append([sound_file, cr_file])

    # do the same for synthetic sounds
    for sound_file in load_sound_file_paths2(synthetic=True):
        cr_file = matlab_outputs / 'synthetic' / f"cr_{sound_file.stem}.mat"
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
        py_out = aud2cor(spect)

        py_out_reordered = np.transpose(py_out, (2, 3, 1, 0))
        
        np.testing.assert_allclose(py_out_reordered, matlab_out, atol=1e-2)
