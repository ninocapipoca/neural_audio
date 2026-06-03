import os
from neural_audio.utils import file_management as fm
def load_sound_files():
    directory = os.path.split(os.path.abspath(__file__))[:-1]
    directory = os.path.join(*directory)
    sound_files = sorted([name for name in os.listdir(directory) if name.endswith('.wav')])
    names = [None] * len(sound_files)
    sounds = [None] * len(sound_files)
    sample_rates = [None] * len(sound_files)

    for s, sound_file in enumerate(sound_files):
        names[s] = sound_file[:-4]
        sounds[s], sample_rates[s] = fm.load_wav(file_path=os.path.join(directory, sound_file))
    
    return names, sounds, sample_rates

def load_sound_file_paths():
    directory = os.path.split(os.path.abspath(__file__))[:-1]
    directory = os.path.join(*directory)
    sound_file_paths = sorted([name for name in os.listdir(directory) if name.endswith('.wav')])
    
    for s, sound_file in enumerate(sound_file_paths):
        sound_file_paths[s] = os.path.join(directory, sound_file)
    
    return sound_file_paths