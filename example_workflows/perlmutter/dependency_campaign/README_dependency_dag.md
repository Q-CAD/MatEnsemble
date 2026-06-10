# MatEnsemble Dependency-DAG Recrystallization Campaign

This folder is an isolated MatEnsemble dependency version of the MACE recrystallization workflow. It keeps the same LAMMPS physics implementation as the parent campaign, but queues each state/seed as a single dependency chain:

```text
prepare amorphous/crystal -> spring calibration -> Frenkel-Ladd -> reversible scaling
```

The dependency is expressed by passing each chore return value into the next chore, following `generic_flux/chores/workflow.py`:

```python
prepared = prepare_amorphous(case_name, seed)
springs = calibrate_springs(case_name, "amorphous", seed, prepared)
fl_ref = run_frenkel_ladd(case_name, "amorphous", seed, prepared, springs)
rs = run_reversible_scaling(case_name, "amorphous", seed, prepared, fl_ref)
```

## Files

- `recryst_mace_matensemble_dag.py`: dependency-aware MatEnsemble workflow.
- `run_recryst_flux_dag.sh`: Podman-HPC/Flux launcher for the DAG workflow.
- `recryst_config_dependency_smoke.json`: tiny one-case smoke configuration.
- `recryst_config_dependency_production_1500_2000_300_600.json`: full production DAG configuration.
- `run_dependency_dag_smoke_interactive.sh`: smoke runner for an existing allocation or `salloc`.
- `submit_dependency_dag_production_1500_2000_300_600.slurm`: production batch script.
- `run_dependency_dag_analysis_1500_2000_300_600.sh`: post-run multi-seed analysis wrapper.

## Smoke Test

Smoke run:

```bash
salloc -N 2 -G 8 -C gpu -A m5064_g --qos interactive -t 00:30:00 \
  bash run_dependency_dag_smoke_interactive.sh
```

Result:

```text
Slurm allocation: 53827013
MatEnsemble workflow: matensemble_workflow-20260603_034429
Chores: Completed=8 Failed=0
Workflow time: 402.4873 s
```

Observed scheduling progression:

```text
Pending=8 Running=0 Completed=0
Pending=6 Running=2 Completed=0   # prep layer
Pending=4 Running=2 Completed=2   # spring layer
Pending=2 Running=2 Completed=4   # FL layer
Pending=0 Running=2 Completed=6   # RS layer
Pending=0 Running=0 Completed=8
```

Smoke report:

```text
reports/dependency_dag_smoke/production_recrystallization_report.md
reports/dependency_dag_smoke/rs_deltaF_summary.csv
reports/dependency_dag_smoke/production_deltaF_300_600.{png,svg,pdf}
```

## Production Submission

Submit the dependency-DAG production campaign with:

```bash
cd /pscratch/sd/s/sbagchi/ORNL_Work/Hetero_interfaces_MACE/FreeEnergy_/matensemble_dependency_campaign
sbatch submit_dependency_dag_production_1500_2000_300_600.slurm
```

This requests 6 GPU nodes. With one MatEnsemble coordinator node, that leaves 5 worker nodes, or 20 one-GPU chores available concurrently.
