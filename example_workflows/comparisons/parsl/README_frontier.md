# Adaptive Parsl Rastrigin Campaign on Frontier

This package contains a self-contained Parsl workflow that models an asynchronous autonomous optimization experiment over the 2D Rastrigin function on `[-5.12, 5.12]^2`.

## Files

- `adaptive_rastrigin_parsl.py` — workflow driver and Parsl app definition.
- `submit_frontier_local.sl` — one-node batch script. Recommended first run.
- `run_frontier_slurm_provider.sh` — login-node driver using Parsl `SlurmProvider` to submit workers.

## Recommended first run: one Frontier node

Edit `submit_frontier_local.sl` and replace `<PROJECT_ID>`.

```bash
sbatch submit_frontier_local.sl
```

This requests one node for 10 minutes and uses 56 Parsl worker slots. Frontier's default core specialization reserves 8 cores, leaving 56 allocatable CPU cores, so this is a good first setting for one single-core worker per task.

## Alternative: Parsl SlurmProvider from the login node

```bash
./run_frontier_slurm_provider.sh <PROJECT_ID>
```

This launches the workflow driver on the login node and lets Parsl submit the worker block with Slurm. Keep the driver lean; do not do heavy computation in the driver.

## Outputs

The workflow writes these files to `--outdir`:

- `evaluations.csv` / `evaluations.json` — one row per completed candidate.
- `best_result.json` — best observed candidate by minimum noisy objective.
- `summary.json` — campaign and Parsl execution summary.
- `generation_events.csv` / `.json` — each task submission.
- `strategy_trace.csv` / `.json` — completion updates and autonomous spawn decisions.
- `parsl_monitoring_summary.json` — best-effort summary of Parsl's `monitoring.db`, if present.

## Useful knobs

- `--n-candidates`: total candidate budget.
- `--target-wall-s`: stop spawning new tasks after this driver wall time; in-flight tasks are allowed to finish.
- `--arm-grid`: number of arms along each dimension.
- `--initial-batch`: broad-coverage startup batch.
- `--max-in-flight`: maximum concurrent Parsl futures.
- `--sleep-scale`: scale simulated wall time down/up for quick tests or longer scheduling behavior.
- `--use-priority/--no-use-priority`: pass a reward-per-cost priority to Parsl when supported by the installed version.

## Interpreting results

The true Rastrigin minimum is at `(0, 0)` with objective `0`. Because this workflow adds noise, the best noisy objective may be slightly below or above the true value. Over time you should see more `local_best` and `thompson_arm` candidates near lower-objective regions, plus continued `global_random` exploration.

In `summary.json`:

- `tasks_scheduled`: number of Parsl app calls submitted.
- `tasks_completed`: number that returned successfully.
- `driver_wall_s`: total workflow driver elapsed time.
- `aggregate_task_actual_wall_s`: sum of worker-side task wall times.
- `effective_parallelism_from_task_wall`: aggregate task wall time divided by driver wall time.
- `worker_slot_occupancy_estimate`: effective parallelism divided by estimated worker slots.
- `app_cpu_utilization_estimate`: app CPU time divided by worker-slot wall time. This is intentionally low because tasks spend most time sleeping.

