from pathlib import Path

def load_sound_file_paths(synthetic: bool=False):

    # uses natural sounds by default
    sound_dir = Path(__file__).resolve().parent

    if synthetic: # if true, use synthetic sounds
        sound_dir = sound_dir / 'synthetic_sounds'

    # returns path objects
    return sorted(p for p in sound_dir.glob('*.wav'))
