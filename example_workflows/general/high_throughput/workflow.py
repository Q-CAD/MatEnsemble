"""Heterogeneous high-throughput MatEnsemble workload.

The workflow deliberately creates a backlog much larger than the Flux
allocation.  Every worker slot therefore has another ready chore available as
soon as its current chore finishes.  Durations are geometrically distributed
and submitted longest-first to reduce the straggler tail.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from matensemble.pipeline import Pipeline


def duration_schedule(
    count: int, minimum: float, maximum: float, seed: int
) -> list[float]:
    """Return deterministic, heterogeneous durations in longest-first order."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("durations require 0 < minimum <= maximum")

    # Stratified log spacing supplies many short chores plus a smaller number
    # of progressively longer chores without relying on lucky random samples.
    ratio = maximum / minimum
    durations = [minimum * ratio ** ((index + 0.5) / count) for index in range(count)]

    # Small deterministic jitter prevents artificial waves of equal-duration
    # completions while keeping every requested duration inside the bounds.
    rng = random.Random(seed)
    durations = [
        min(maximum, max(minimum, value * rng.uniform(0.90, 1.10)))
        for value in durations
    ]
    return sorted(durations, reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=int,
        default=512,
        help="number of independent chores to enqueue (default: 512)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.25,
        help="shortest target task duration (default: 0.25)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="longest target task duration (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=("sleep", "cpu"),
        default="sleep",
        help="sleep simulates occupancy; cpu performs busy floating-point work",
    )
    parser.add_argument(
        "--basedir",
        type=Path,
        default=Path.cwd(),
        help="parent directory for the timestamped workflow directory",
    )
    parser.add_argument(
        "--buffer-time",
        type=float,
        default=0.05,
        help="future-processing poll interval in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--log-delay",
        type=float,
        default=1.0,
        help="status logging interval in seconds (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    durations = duration_schedule(
        args.tasks, args.min_seconds, args.max_seconds, args.seed
    )
    pipe = Pipeline(basedir=str(args.basedir))

    @pipe.chore(name="heterogeneous-work", num_tasks=1, cores_per_task=1)
    def heterogeneous_work(
        task_id: int, target_seconds: float, mode: str
    ) -> dict[str, float | int | str]:
        """Occupy one Flux worker slot for approximately ``target_seconds``."""
        started = time.monotonic()
        iterations = 0
        checksum = 0.0

        if mode == "sleep":
            time.sleep(target_seconds)
        else:
            deadline = started + target_seconds
            value = task_id + 1.0
            while time.monotonic() < deadline:
                # Bounded batches reduce timer overhead while doing real CPU work.
                for _ in range(10_000):
                    value = math.sin(value) ** 2 + math.cos(value) ** 2 + 0.000001
                checksum += value
                iterations += 10_000

        return {
            "task_id": task_id,
            "target_seconds": target_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "mode": mode,
            "iterations": iterations,
            "checksum": checksum,
        }

    @pipe.chore(name="summarize", num_tasks=1, cores_per_task=1)
    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        elapsed = [float(record["elapsed_seconds"]) for record in records]
        return {
            "tasks": len(records),
            "minimum_elapsed_seconds": min(elapsed),
            "maximum_elapsed_seconds": max(elapsed),
            "total_task_seconds": sum(elapsed),
            "mode": records[0]["mode"],
        }

    # Longest-processing-time-first ordering minimizes the final straggler tail.
    # Because all chores are independent and one-core, the ready queue remains
    # deep until the campaign is almost complete.
    outputs = [
        heterogeneous_work(task_id, duration, args.mode)
        for task_id, duration in enumerate(durations)
    ]
    summary = summarize(outputs)

    workflow_future = pipe.submit(
        adaptive=True,
        buffer_time=args.buffer_time,
        log_delay=args.log_delay,
        set_cpu_affinity=True,
    )
    results = workflow_future.result()
    final_summary = results[summary.chore_id]
    print(final_summary)
    return final_summary


if __name__ == "__main__":
    main()
