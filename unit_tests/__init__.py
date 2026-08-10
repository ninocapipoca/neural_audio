"""Unit tests for the neural_audio toolbox.

Puts ``src`` on ``sys.path`` so the tests import ``neural_audio`` straight from the
working tree. There is no editable install in this project, and this package
``__init__`` is imported before either test module during unittest discovery, so
this is what lets VS Code (and a plain ``python -m unittest``) collect the tests
without any PYTHONPATH set up by hand.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
