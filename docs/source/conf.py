# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "MatEnsemble"
copyright = "2026, Soumendu Bagchi, Kaleb Duchesneau"
author = "Soumendu Bagchi, Kaleb Duchesneau"
release = "v0.1.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
    "inherited-members": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "flux",
    "flux.job",
    "mpi4py",
    "mpi4py.MPI",
    "lammps",
    "numpy",
    "pandas",
    "torch",
    "scipy",
    "sklearn",
    "matplotlib",
    "matplotlib.pyplot",
    "ase",
    "pymatgen",
    "seaborn",
    "ovito",
    "ovito.io",
    "ovito.data",
    "ovito.pipeline",
    "redis",
]

html_theme = "sphinx_rtd_theme"
