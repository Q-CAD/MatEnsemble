================
Curated Examples
================

Curated examples live under ``example_workflows/`` and are indexed by
``example_workflows/examples.toml``. The MCP server uses that manifest as its
source of context.

Examples included in MCP context
================================

* ``generic_flux/chores`` — portable, site-independent dependency-aware Python chores.
* ``generic_flux/mpi`` — portable, site-independent MPI-enabled Python chores.
* ``generic_flux/strategy`` — portable, site-independent adaptive strategy and ``ChoreSpec``.
* ``generic_flux/executable`` — portable, site-independent executable chores using ``Pipeline.exec``.
* ``frontier/lammps_smoke`` — Frontier LAMMPS GPU smoke workflow and CLI batch scripts.
* ``perlmutter/lammps_smoke`` — Perlmutter LAMMPS GPU smoke workflow and CLI batch scripts.
* ``perlmutter/lammps_mace`` — Perlmutter LAMMPS/MACE workflow and launch pattern.
* ``perlmutter/dependency_campaign`` — dependency-aware recrystallization campaign and smoke config.

The ``generic_flux`` examples show the Python workflow shape. They are intended
to be adapted to Frontier, Perlmutter, Pathfinder, Linux containers, or another
Flux-capable runtime by pairing them with the appropriate system-specific
launch scripts, containers, scheduler flags, and dependency setup.

Examples excluded from MCP context
==================================

The MCP manifest intentionally excludes generated workflow output directories,
large model binaries, pickled registry artifacts, raw stdout/stderr logs, and
notebooks. Keep those artifacts out of ``examples.toml`` unless they are reduced
to small text examples.
