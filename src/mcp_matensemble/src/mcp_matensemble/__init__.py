"""MCP server helpers for MatEnsemble."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("mcp-matensemble")
except PackageNotFoundError:
    __version__ = "0+unknown"
