# Adaptive LAMMPS Smoke Campaign

This example runs a tiny Lennard-Jones LAMMPS campaign through MatEnsemble.
It is meant for testing the science workflow shape, not for production
materials conclusions.

The workflow demonstrates:

- a resource-aware LAMMPS simulation chore,
- a dependent analysis chore that receives the LAMMPS result through an
  `OutputReference`,
- a user strategy that spawns validation LAMMPS runs for promising screen
  results while the workflow is still running.

Run it inside the repository dev container or another Flux-capable environment
with the LAMMPS Python module installed:

```bash
flux start -s 2 python example_workflows/generic/lammps_adaptive/workflow.py
```

The script writes normal MatEnsemble output under a timestamped
`matensemble_workflow-*` directory. Validation LAMMPS chores also write
`final.data` and `final.dump` in their per-chore work directories for quick
inspection.
