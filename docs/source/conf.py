# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _version

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'neural_audio'
copyright = '2026, CLNM'
author = 'C L Nina Matos'

# Make sure version stays up to date
try:
    release = _version('neural_audio')
except PackageNotFoundError:
    release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Sphinx extension modules
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Show function names cleanly
add_module_names = False
toc_object_entries_show_parents = 'hide'

# Render default values as they are written in the source (np.complex64) rather their repr (<class 'numpy.complex64'>)
autodoc_preserve_defaults = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
