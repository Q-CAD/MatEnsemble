"""Standalone, read-only MatEnsemble workflow dashboard."""

from .app import create_dashboard_app
from .discovery import WorkflowCatalog

__all__ = ["WorkflowCatalog", "create_dashboard_app"]
