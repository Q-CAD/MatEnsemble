#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv-docs}"
DOCS_DIR="${DOCS_DIR:-docs}"

# 1) Create venv (idempotent)
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

# 2) Install doc build deps + core runtime deps required for imports
python -m pip install -U \
  numpy \
  sphinx \
  sphinx-rtd-theme \
  myst-parser \
  sphinx-autodoc-typehints

# Optional: if you want to try fully documenting dynopro/redis too, uncomment:
# python -m pip install -U pandas mpi4py

# 3) Ensure strategy is a package (warn, don’t modify)
if [ ! -f "matensemble/strategy/__init__.py" ]; then
  echo "ERROR: matensemble/strategy/__init__.py is missing."
  echo "Create it so Sphinx can discover the strategy subpackage."
  exit 1
fi

# 4) Find (or create) Sphinx conf.py
DEFAULT_SOURCE="$DOCS_DIR/source"
DEFAULT_BUILD="$DOCS_DIR/build"
CONF_DIR=""
SOURCE_DIR=""

if [ -f "$DEFAULT_SOURCE/conf.py" ]; then
  CONF_DIR="$DEFAULT_SOURCE"
  SOURCE_DIR="$DEFAULT_SOURCE"
elif [ -f "$DOCS_DIR/conf.py" ]; then
  CONF_DIR="$DOCS_DIR"
  SOURCE_DIR="$DOCS_DIR"
else
  CONF_PY="$(find "$DOCS_DIR" -maxdepth 3 -name conf.py -print -quit 2>/dev/null || true)"
  if [ -n "${CONF_PY:-}" ]; then
    CONF_DIR="$(dirname "$CONF_PY")"
    SOURCE_DIR="$CONF_DIR"
  else
    echo "No conf.py found under '$DOCS_DIR/'. Initializing minimal Sphinx config in '$DEFAULT_SOURCE'..."
    mkdir -p "$DEFAULT_SOURCE/_static" "$DEFAULT_SOURCE/_templates" "$DEFAULT_BUILD"

    PROJECT_NAME="${PROJECT_NAME:-MatEnsemble}"
    AUTHOR_NAME="${AUTHOR_NAME:-$(git config user.name 2>/dev/null || true)}"
    AUTHOR_NAME="${AUTHOR_NAME:-Unknown}"

    cat > "$DEFAULT_SOURCE/conf.py" <<PY
from __future__ import annotations

import sys
from pathlib import Path

# Make project importable regardless of whether conf.py lives in docs/ or docs/source/
HERE = Path(__file__).resolve()
for parent in HERE.parents:
    if (parent / "matensemble").is_dir():
        sys.path.insert(0, str(parent))
        break

project = "${PROJECT_NAME}"
author = "${AUTHOR_NAME}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# If you don't have Flux bindings installed locally, mock them so autodoc can import.
autodoc_mock_imports = [
    "flux",
    "flux.job",
    "mpi4py",
    "redis",
    "pandas",
]

autosummary_generate = True
root_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
PY

    # Minimal root doc so build succeeds
    cat > "$DEFAULT_SOURCE/index.rst" <<'RST'
MatEnsemble Documentation
========================

.. toctree::
   :maxdepth: 2
   :caption: API

   api/modules
RST

    CONF_DIR="$DEFAULT_SOURCE"
    SOURCE_DIR="$DEFAULT_SOURCE"
  fi
fi

# 5) Regenerate API stubs into the *actual* source dir
API_DIR="$SOURCE_DIR/api"
rm -rf "$API_DIR"
sphinx-apidoc -f -o "$API_DIR" matensemble

# 6) Build docs
OUT_DIR="$DEFAULT_BUILD/html"
mkdir -p "$OUT_DIR"

# -W: treat warnings as errors (add later once docstrings are clean)
# --keep-going: show all issues in one run
sphinx-build -b html --keep-going -c "$CONF_DIR" "$SOURCE_DIR" "$OUT_DIR"

echo
echo "Docs built: $ROOT/$OUT_DIR/index.html"
