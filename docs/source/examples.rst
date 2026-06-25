===================
Repository Examples
===================

MatEnsemble is designed to be intuitive so if you are already familiar with HPC and
want to get from 0 to 60 as fast as possible then take a look at the example workflows
that we have in the repository. There are explanations for anything that may be confusing.

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

