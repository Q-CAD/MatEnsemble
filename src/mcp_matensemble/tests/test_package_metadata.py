from __future__ import annotations

import tomllib
import importlib.metadata
from pathlib import Path

import mcp_matensemble


def test_mcp_version_matches_matensemble_version():
    root = Path(__file__).resolve().parents[3]
    core = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mcp = tomllib.loads(
        (root / "src" / "mcp_matensemble" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert mcp["project"]["version"] == core["project"]["version"]


def test_public_version_comes_from_package_metadata():
    assert mcp_matensemble.__version__ == importlib.metadata.version(
        "mcp-matensemble"
    )
