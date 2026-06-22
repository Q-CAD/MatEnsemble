# Contributing to MatEnsemble

Thanks for helping improve MatEnsemble. The project sits at an awkward but interesting boundary between Python workflow APIs, Flux scheduling, containers, and site-specific HPC launch details, so good issues and small reproducible examples are especially valuable.

## Questions and Bugs

Before opening an issue, please check the documentation and existing issues. When reporting a bug, include:

- MatEnsemble version and install method
- HPC system or local runtime, such as Frontier, Perlmutter, Pathfinder, conda, or a Linux container
- Container tag or SIF path, if applicable
- Flux, Slurm, MPI, and Python versions when known
- The smallest workflow script that reproduces the problem
- Relevant `matensemble_workflow.log`, `status.json`, and per-chore `stderr` snippets

Do not post secrets, tokens, private allocations, or sensitive cluster paths in public issues. For security-sensitive reports, contact the maintainers listed in `pyproject.toml`.

## Development Setup

Use Python 3.12 and `uv`:

```bash
uv sync --dev
uv run pytest -q
```

Build docs locally before changing public APIs or examples:

```bash
uv run sphinx-build -b html -W docs/source /tmp/matensemble-docs-build
```

## Pull Requests

Keep changes focused. A good MatEnsemble pull request usually includes:

- tests for runtime behavior or script generation changes
- docs updates for user-facing behavior
- example updates when a site launch pattern changes
- notes about which systems were actually tested

Avoid committing generated workflow output directories, large model binaries, local `.DS_Store` files, or cluster-specific absolute paths.

## Style

Use clear, typed Python where practical. Docstrings should follow NumPy or Google style consistently within a file. Shell scripts should use `set -euo pipefail` unless there is a specific reason not to.
