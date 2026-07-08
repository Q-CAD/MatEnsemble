These workflows are site-independent MatEnsemble examples. They show the Python workflow shape;
use system-specific examples for launcher scripts, containers, scheduler flags, and science
dependency setup.

The examples in this directory demonstrate core MatEnsemble workflow patterns that are portable
across Frontier, Perlmutter, Pathfinder, Linux containers, and any other Flux-capable runtime.

Use these examples to understand how to structure `Pipeline` objects, chores, executable tasks,
MPI chores, dependencies, and adaptive strategies. When running on a specific system, pair the
Python workflow pattern here with that system's launch instructions and examples.

MatEnsemble workflows are centered around `Chores` which are delayed function calls that are serialized and later called on compute resources that MatEnsemble has control over

Available examples:

- `dependencies`: dependency-aware Python chores using `OutputReference`.
- `executable`: external command chores through `Pipeline.exec`.
- `mpi`: portable MPI-enabled Python chores.
- `strategy`: adaptive `ChoreSpec` spawning.
- `lammps_adaptive`: a tiny LAMMPS Python-module campaign with adaptive validation.

## Dev container and local Flux runs

Inside the repository dev container, run these examples with a multi-rank Flux test instance:

```bash
flux start -s 2 python example_workflows/generic/dependencies/workflow.py
```

The dev container also sets:

```bash
MATENSEMBLE_FLUX_START="flux start -s 2"
```

so this works in an interactive shell:

```bash
$MATENSEMBLE_FLUX_START python example_workflows/generic/dependencies/workflow.py
```

MatEnsemble drains broker rank `0` before measuring available resources. A one-rank local `flux start` session therefore leaves no usable rank for chores, even though `flux resource list` itself works.
