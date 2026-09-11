from pathlib import Path

def load_sound_file_paths(synthetic: bool = False) -> list[Path]:
    """Return the paths of provided example sounds.

    :param synthetic: If ``True``, return the synthetic stimuli (ripples) instead of natural sounds.
    :type synthetic: bool, optional, default=False

    :returns: The ``.wav`` files in the requested set, sorted by filename.
    :rtype: list of pathlib.Path
    """

    sound_dir = Path(__file__).resolve().parent

    if synthetic:
        sound_dir = sound_dir / 'synthetic_sounds'

    return sorted(p for p in sound_dir.glob('*.wav'))
