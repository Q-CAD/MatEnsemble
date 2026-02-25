"""
Pipeline frontend for MatEnsemble.

Exports the user-facing Pipeline API for building and running DAG workflows that
compile to Flux jobs.
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
