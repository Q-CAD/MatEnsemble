===================
Repository Examples
===================

Examples live under ``example_workflows/``. The MCP server discovers and reads
the files directly from ``example_workflows/<system>/`` at request time, so
changes to those files are immediately reflected in MCP responses.

Example directories
===================

* ``generic/dependencies`` — portable, site-independent dependency-aware Python chores.
* ``generic/mpi`` — portable, site-independent MPI-enabled Python chores.
* ``generic/strategy`` — portable, site-independent adaptive strategy and ``ChoreSpec``.
* ``generic/executable`` — portable, site-independent executable chores using ``Pipeline.exec``.
* ``frontier/lammps_smoke`` — Frontier LAMMPS GPU smoke workflow and CLI batch scripts.
* ``pathfinder/lammps_smoke`` — Pathfinder LAMMPS CPU smoke workflow and CLI batch scripts.
* ``perlmutter/lammps_smoke`` — Perlmutter LAMMPS GPU smoke workflow and CLI batch scripts.
* ``perlmutter/lammps_mace`` — Perlmutter LAMMPS/MACE workflow and launch pattern.
* ``perlmutter/dependency_campaign`` — dependency-aware recrystallization campaign and smoke config.

The ``generic`` examples show the Python workflow shape. They are intended
to be adapted to Frontier, Perlmutter, Pathfinder, Linux containers, or another
Flux-capable runtime by pairing them with the appropriate system-specific
launch scripts, containers, scheduler flags, and dependency setup.

MCP loading behavior
====================

``get_examples(system)`` always returns the portable files under
``example_workflows/generic`` followed by every file under the matching system
tree. This ensures an agent has the canonical MatEnsemble workflow patterns as
well as the site-specific launch and runtime details.
``get_example(system, name)`` returns every file under one example directory.
The generic Linux/Flux examples are stored under ``example_workflows/generic``.

Keep generated workflow outputs, model binaries, pickled artifacts, and raw
logs outside these source example directories unless they are intentionally
part of the context supplied to MCP clients.
