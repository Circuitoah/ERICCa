import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project   = "ERICCa"
copyright = "2026, A. J. Smith, K. Godbey, C. Hebborn"
author    = "A. J. Smith, K. Godbey, C. Hebborn"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme = "furo"

napoleon_google_docstring = False
napoleon_numpy_docstring  = True
