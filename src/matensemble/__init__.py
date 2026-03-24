"""
MatEnsemble

MatEnsemble is a Python workflow library for building and running
high-throughput and dependency-aware workflows on HPC systems.
It lets users define delayed Python and executable jobs, connect them
through data dependencies, and submit them for execution with Flux.
"""

__author__ = ["Soumendu Bagchi", "Kaleb Duchesneau"]
__package__ = "matensemble"

# Re-export core data model types at the package root for convenience and for
# backwards compatibility with code/tests that import from `matensemble`.
from .model import OutputReference, Resources, JobFlavor  # noqa: E402,F401
