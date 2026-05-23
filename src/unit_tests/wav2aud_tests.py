import unittest
from parameterized import parameterized
import sys
from pathlib import Path
import numpy as np
import soundfile as soundf
from neural_audio.wav2aud_zpk import wav2aud

test_dir = Path(__file__).parent # unit_tests folder

sf = 16000 # sampling freq

def construct_cases_audio():
    sound_dir = test_dir / 'test_sounds'

    # eg 'audio.wav'.stem is 'audio'
    out = []
    for sound in sound_dir.glob('*.wav'):
        csv = (test_dir / 'matlab_outputs' / sound.stem).with_suffix('.csv')

        out.append([sound, csv])

    return out

def name_audio(f, n, p):
    return f"test_audio_{str(p[0][0].stem)}"

def construct_cases_sinusoids():
    duration = 2
    out = []

    for freq in np.arange(20, 20000, 2002): # human hearing range. test 10 evenly spaced frequency values    
        t = np.arange(0, duration + 1/sf, 1/sf)  
        wave = np.sin(2 * np.pi * freq * t[:32000])
        out.append([wave, freq])
    
    return out

def name_sinusoids(f, n, p):
    return f"test_{int(p[0][1])}Hz"

audio_cases = construct_cases_audio()
sin_cases = construct_cases_sinusoids()

class audioTests(unittest.TestCase):
    @parameterized.expand(audio_cases, name_func=name_audio)
    def test_audiofiles(self, sound, csv):
        soundData, sample_rate = soundf.read(sound)

        matlab_out = np.genfromtxt(csv, delimiter=',')
        py_out = wav2aud(soundData)

        #match = np.isclose(matlab_out, py_out, rtol=1e-5)
        #percent_match = (np.sum(match) / matlab_out.size) * 100

        #self.assertGreaterEqual(percent_match, 99)
        np.testing.assert_allclose(py_out, matlab_out, atol=1e-2)

class sinusoidTests(unittest.TestCase):
    @parameterized.expand(sin_cases, name_func=name_sinusoids)
    def test_sinusoid(self, wave, freq):
        file = str(freq) + 'hz.csv'
        csv = test_dir / 'matlab_outputs' / file

        matlab_out = np.genfromtxt(csv, delimiter=',')
        py_out = wav2aud(wave)

        #match = np.isclose(matlab_out, py_out, rtol=1e-5)
        #percent_match = (np.sum(match) / matlab_out.size) * 100

        #self.assertGreaterEqual(percent_match, 99)
        diff = np.abs(py_out - matlab_out)
        np.testing.assert_allclose(0 * diff, diff, atol=1e-2)

class stepTest(unittest.TestCase):
    def test_stepfunc(self):
        step_input = np.ones(500)
        csv = test_dir / 'matlab_outputs' / 'step.csv'

        matlab_out = np.genfromtxt(csv, delimiter=',')
        py_out = wav2aud(step_input)

        #match = np.isclose(matlab_out, py_out, rtol=1e-5)
        #percent_match = (np.sum(match) / matlab_out.size) * 100

        #self.assertGreaterEqual(percent_match, 99)
        diff = np.abs(py_out - matlab_out)
        np.testing.assert_allclose(0 * diff, diff, atol=1e-2)


    