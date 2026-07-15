from pathlib import Path

def load_sound_file_paths2(): # using pathlib instead
    directory = Path(__file__).resolve().parent

    # returns path objects
    return sorted(p for p in directory.glob('*.wav'))

import os

def load_sound_file_paths():
    directory = os.path.split(os.path.abspath(__file__))[:-1]
    directory = os.path.join(*directory)
    sound_file_paths = sorted([name for name in os.listdir(directory) if name.endswith('.wav')])
    
    for s, sound_file in enumerate(sound_file_paths):
        sound_file_paths[s] = os.path.join(directory, sound_file)
    
    return sound_file_paths