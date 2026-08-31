# Heterogeneous high-throughput workflow

This example enqueues hundreds of independent, single-core chores with target
durations spread geometrically between a configurable minimum and maximum. The
ready backlog is intentionally much larger than a typical allocation, allowing
MatEnsemble and Flux to immediately refill worker slots as chores finish.

Run it from a Flux allocation with at least two ranks (MatEnsemble reserves rank
0 for the broker):

```bash
flux start -s 5 python example_workflows/general/high_throughput/workflow.py \
  --tasks 512 --min-seconds 0.25 --max-seconds 20
```

The default `sleep` mode is a scheduler-throughput demonstration: chores occupy
allocated slots but intentionally do not consume CPU while sleeping. Use `cpu`
mode to keep the allocated CPU cores busy with synthetic floating-point work:

```bash
flux start -s 5 python example_workflows/general/high_throughput/workflow.py \
  --tasks 512 --min-seconds 0.25 --max-seconds 20 --mode cpu
```

For sustained saturation, set `--tasks` to at least 4–10 times the usable core
count. Tasks are inserted longest-first to reduce the straggler tail, while a
final summary chore depends on every worker chore and reports observed runtime.

Useful controls:

- `--tasks`: total number of independent worker chores.
- `--min-seconds` and `--max-seconds`: target runtime range.
- `--mode sleep|cpu`: simulated occupancy or active CPU work.
- `--buffer-time`: how frequently completed futures are processed; the default
  `0.05` seconds favors quick slot refill for short chores.
- `--basedir`: parent of the timestamped `matensemble_workflow-*` directory.
