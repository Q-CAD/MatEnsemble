#!/usr/bin/env python3
"""
Adaptive asynchronous Parsl campaign over the 2D Rastrigin function.

This is designed for Frontier-style many-task workflow experiments.  Each Parsl
Python app evaluates one candidate point, sleeps for a heterogeneous simulated
wall time, computes a noisy objective, and returns a structured result.  The
workflow manager runs an autonomous asynchronous loop: whenever a task completes,
it updates per-arm statistics and optionally spawns one more candidate until the
candidate or wall-clock budget is reached.

Outputs are written to --outdir:
  evaluations.csv / evaluations.json
  best_result.json
  summary.json
  generation_events.csv / generation_events.json
  strategy_trace.csv / strategy_trace.json
  parsl_monitoring_summary.json, when a monitoring.db file is found
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname, address_by_interface

LOW, HIGH = -5.12, 5.12
DOMAIN_WIDTH = HIGH - LOW


@python_app
def evaluate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one noisy Rastrigin candidate after simulated wall-time delay."""
    import math
    import os
    import platform
    import random
    import time

    rng = random.Random(candidate["task_seed"])
    expected_wall_s = float(candidate["expected_wall_s"])
    jitter = rng.uniform(0.85, 1.20)
    sleep_s = max(0.0, expected_wall_s * jitter)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    time.sleep(sleep_s)

    x = float(candidate["x"])
    y = float(candidate["y"])
    objective_clean = 20.0 + (x * x - 10.0 * math.cos(2.0 * math.pi * x)) + (y * y - 10.0 * math.cos(2.0 * math.pi * y))
    noise = rng.gauss(0.0, float(candidate["noise_sigma"]))
    objective = objective_clean + noise
    reward = -objective

    actual_wall_s = time.perf_counter() - t0_wall
    cpu_s = time.process_time() - t0_cpu

    return {
        "candidate_id": int(candidate["candidate_id"]),
        "x": x,
        "y": y,
        "arm_ix": int(candidate["arm_ix"]),
        "arm_iy": int(candidate["arm_iy"]),
        "arm_id": int(candidate["arm_id"]),
        "grid_arm": candidate["grid_arm"],
        "expected_wall_s": expected_wall_s,
        "actual_wall_s": actual_wall_s,
        "cpu_s": cpu_s,
        "objective_clean": objective_clean,
        "noise": noise,
        "objective": objective,
        "reward": reward,
        "source": candidate["source"],
        "priority": candidate.get("priority"),
        "priority_score": candidate.get("priority_score"),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "parsl_worker_rank": os.environ.get("PARSL_WORKER_RANK", ""),
        "parsl_worker_count": os.environ.get("PARSL_WORKER_COUNT", ""),
        "parsl_worker_pool_id": os.environ.get("PARSL_WORKER_POOL_ID", ""),
        "status": "completed",
    }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def clamp(v: float, lo: float = LOW, hi: float = HIGH) -> float:
    return max(lo, min(hi, v))


def arm_for_xy(x: float, y: float, arm_grid: int) -> Tuple[int, int, int, str]:
    ix = int((x - LOW) / DOMAIN_WIDTH * arm_grid)
    iy = int((y - LOW) / DOMAIN_WIDTH * arm_grid)
    ix = max(0, min(arm_grid - 1, ix))
    iy = max(0, min(arm_grid - 1, iy))
    arm_id = iy * arm_grid + ix
    return ix, iy, arm_id, f"{ix},{iy}"


def arm_bounds(arm_id: int, arm_grid: int) -> Tuple[float, float, float, float]:
    ix = arm_id % arm_grid
    iy = arm_id // arm_grid
    dx = DOMAIN_WIDTH / arm_grid
    return LOW + ix * dx, LOW + (ix + 1) * dx, LOW + iy * dx, LOW + (iy + 1) * dx


def sample_in_arm(rng: random.Random, arm_id: int, arm_grid: int) -> Tuple[float, float]:
    x0, x1, y0, y1 = arm_bounds(arm_id, arm_grid)
    return rng.uniform(x0, x1), rng.uniform(y0, y1)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_config(args: argparse.Namespace) -> Config:
    """Create a Parsl config for local HTEx or Frontier SlurmProvider HTEx."""
    run_dir = str(Path(args.run_dir).resolve())
    monitoring = None

    if args.monitoring:
        try:
            from parsl.monitoring.monitoring import MonitoringHub

            if args.parsl_config == "frontier-slurm":
                hub_address = address_by_interface(args.frontier_interface)
            else:
                hub_address = address_by_hostname()
            monitoring = MonitoringHub(
                hub_address=hub_address,
                hub_port=args.monitoring_port,
                resource_monitoring_interval=args.resource_monitoring_interval,
                monitoring_debug=False,
            )
        except Exception as exc:  # monitoring extras might not be installed
            print(f"[warn] Parsl monitoring disabled: {exc}", file=sys.stderr)
            monitoring = None

    if args.parsl_config == "local-htex":
        # Intended for a one-node batch allocation or quick local testing.
        executor = HighThroughputExecutor(
            label="htex",
            address=address_by_hostname(),
            cores_per_worker=args.cores_per_worker,
            max_workers_per_node=args.max_workers_per_node,
            provider=LocalProvider(init_blocks=1, max_blocks=1),
        )
    elif args.parsl_config == "frontier-slurm":
        if not args.account:
            raise ValueError("--account is required with --parsl-config frontier-slurm")
        scheduler_options = [f"#SBATCH -A {args.account}"]
        if args.job_name:
            scheduler_options.append(f"#SBATCH -J {args.job_name}")
        if args.core_spec is not None:
            scheduler_options.append(f"#SBATCH -S {args.core_spec}")
        if args.extra_sbatch:
            scheduler_options.extend(args.extra_sbatch)

        worker_init = args.worker_init or "module load parsl/2024.12.2"
        executor = HighThroughputExecutor(
            label="htex",
            address=address_by_interface(args.frontier_interface),
            cores_per_worker=args.cores_per_worker,
            max_workers_per_node=args.max_workers_per_node,
            provider=SlurmProvider(
                cmd_timeout=60,
                nodes_per_block=args.nodes_per_block,
                init_blocks=args.init_blocks,
                min_blocks=0,
                max_blocks=args.max_blocks,
                partition=args.partition,
                scheduler_options="\n".join(scheduler_options),
                worker_init=worker_init,
                walltime=args.block_walltime,
                launcher=SrunLauncher(),
            ),
        )
    else:
        raise ValueError(f"Unknown --parsl-config {args.parsl_config}")

    return Config(executors=[executor], monitoring=monitoring, run_dir=run_dir, strategy=None)


def initial_coverage_candidates(args: argparse.Namespace, rng: random.Random, start_id: int) -> List[Dict[str, Any]]:
    n = min(args.initial_batch, args.n_candidates)
    # Latin-hypercube-like coverage across the square, randomized independently in x/y.
    xs = list(range(n))
    ys = list(range(n))
    rng.shuffle(xs)
    rng.shuffle(ys)
    candidates = []
    for i in range(n):
        x = LOW + ((xs[i] + rng.random()) / n) * DOMAIN_WIDTH
        y = LOW + ((ys[i] + rng.random()) / n) * DOMAIN_WIDTH
        candidates.append(make_candidate(args, rng, start_id + i, "initial_coverage", x=x, y=y))
    return candidates


def thompson_arm(args: argparse.Namespace, rng: random.Random, arm_stats: Dict[int, Dict[str, float]]) -> Tuple[int, Dict[str, float]]:
    best_arm = 0
    best_draw = -float("inf")
    draws: Dict[int, float] = {}
    for arm_id in range(args.arm_grid * args.arm_grid):
        count = int(arm_stats[arm_id]["count"])
        mean = float(arm_stats[arm_id]["mean_reward"]) if count else args.prior_mean_reward
        sigma = args.prior_reward_scale / math.sqrt(count + 1.0)
        draw = rng.gauss(mean, sigma)
        draws[arm_id] = draw
        if draw > best_draw:
            best_draw = draw
            best_arm = arm_id
    trace = {"selected_arm": best_arm, "selected_draw": best_draw, "max_draw": best_draw}
    return best_arm, trace


def choose_source(args: argparse.Namespace, rng: random.Random, best: Optional[Dict[str, Any]]) -> str:
    p_random = args.p_random
    p_local = args.p_local if best is not None else 0.0
    p_thompson = args.p_thompson
    total = p_random + p_local + p_thompson
    u = rng.random() * total
    if u < p_random:
        return "global_random"
    if u < p_random + p_local:
        return "local_best"
    return "thompson_arm"


def expected_wall(args: argparse.Namespace, rng: random.Random, source: str, arm_id: int) -> float:
    # Heterogeneous runtimes: lognormal base, source/arm modifiers, and occasional stragglers.
    if source == "local_best":
        base = rng.lognormvariate(math.log(2.0), 0.35)
    elif source == "thompson_arm":
        base = rng.lognormvariate(math.log(3.0), 0.55)
    elif source == "initial_coverage":
        base = rng.uniform(1.0, 8.0)
    else:
        base = rng.lognormvariate(math.log(4.0), 0.75)

    # Mild deterministic arm heterogeneity, so some grid areas are more expensive.
    arm_modifier = 0.8 + 0.4 * ((arm_id % max(1, args.arm_grid)) / max(1, args.arm_grid - 1))
    if rng.random() < args.straggler_probability:
        base *= rng.uniform(2.0, 4.0)
    return max(args.min_wall_s, min(args.max_wall_s, base * arm_modifier * args.sleep_scale))


def priority_for_candidate(args: argparse.Namespace, arm_stats: Dict[int, Dict[str, float]], arm_id: int, expected_s: float, source: str) -> Tuple[float, float]:
    count = int(arm_stats[arm_id]["count"])
    mean_reward = float(arm_stats[arm_id]["mean_reward"]) if count else args.prior_mean_reward

    # Shift reward to positive utility for cost ranking. Larger score is better.
    # HTEx priority convention: lower numeric priority runs earlier.
    source_bonus = {"local_best": 8.0, "thompson_arm": 5.0, "global_random": 2.0, "initial_coverage": 0.0}.get(source, 0.0)
    priority_score = (mean_reward + args.priority_reward_shift + source_bonus) / max(0.001, expected_s)
    htex_priority = -priority_score
    return htex_priority, priority_score


def make_candidate(
    args: argparse.Namespace,
    rng: random.Random,
    candidate_id: int,
    source: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    best: Optional[Dict[str, Any]] = None,
    arm_stats: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    thompson_trace: Dict[str, Any] = {}
    if x is None or y is None:
        if source == "local_best" and best is not None:
            step = DOMAIN_WIDTH / max(2.0, args.arm_grid * 1.5)
            x = clamp(rng.gauss(float(best["x"]), step))
            y = clamp(rng.gauss(float(best["y"]), step))
        elif source == "thompson_arm" and arm_stats is not None:
            arm_id, thompson_trace = thompson_arm(args, rng, arm_stats)
            x, y = sample_in_arm(rng, arm_id, args.arm_grid)
        else:
            x = rng.uniform(LOW, HIGH)
            y = rng.uniform(LOW, HIGH)

    arm_ix, arm_iy, arm_id, grid_arm = arm_for_xy(float(x), float(y), args.arm_grid)
    expected = expected_wall(args, rng, source, arm_id)
    if arm_stats is None:
        # During initial generation arm_stats exists but is uninformative; this fallback keeps the function reusable.
        arm_stats = {i: {"count": 0.0, "mean_reward": 0.0} for i in range(args.arm_grid * args.arm_grid)}
    priority, priority_score = priority_for_candidate(args, arm_stats, arm_id, expected, source)
    return {
        "candidate_id": candidate_id,
        "x": float(x),
        "y": float(y),
        "arm_ix": arm_ix,
        "arm_iy": arm_iy,
        "arm_id": arm_id,
        "grid_arm": grid_arm,
        "expected_wall_s": expected,
        "source": source,
        "task_seed": rng.randrange(1, 2**31 - 1),
        "noise_sigma": args.noise_sigma,
        "priority": priority,
        "priority_score": priority_score,
        "thompson_trace": thompson_trace,
    }


def update_arm_stats(arm_stats: Dict[int, Dict[str, float]], result: Dict[str, Any]) -> None:
    arm_id = int(result["arm_id"])
    reward = float(result["reward"])
    s = arm_stats[arm_id]
    count = int(s["count"])
    new_count = count + 1
    old_mean = float(s["mean_reward"])
    s["count"] = new_count
    s["mean_reward"] = old_mean + (reward - old_mean) / new_count
    s["sum_reward"] = float(s.get("sum_reward", 0.0)) + reward


def summarize_monitoring_db(search_roots: Iterable[Path]) -> Dict[str, Any]:
    dbs: List[Path] = []
    for root in search_roots:
        if root.exists():
            dbs.extend(root.rglob("monitoring.db"))
    if not dbs:
        return {"found": False, "note": "No monitoring.db found."}

    db_path = max(dbs, key=lambda p: p.stat().st_mtime)
    summary: Dict[str, Any] = {"found": True, "path": str(db_path), "tables": {}, "best_effort": True}
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        table_names = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for table in table_names:
            entry: Dict[str, Any] = {}
            try:
                entry["row_count"] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                entry["columns"] = cols
                if "task_status_name" in cols:
                    entry["task_status_counts"] = dict(cur.execute(f"SELECT task_status_name, COUNT(*) FROM {table} GROUP BY task_status_name").fetchall())
                if "task_func_name" in cols:
                    entry["task_func_counts"] = dict(cur.execute(f"SELECT task_func_name, COUNT(*) FROM {table} GROUP BY task_func_name").fetchall())
            except Exception as exc:
                entry["error"] = str(exc)
            summary["tables"][table] = entry
        con.close()
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def build_summary(
    args: argparse.Namespace,
    evaluations: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    generation_events: List[Dict[str, Any]],
    strategy_trace: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    t_start: float,
    t_end: float,
) -> Dict[str, Any]:
    actuals = [float(r["actual_wall_s"]) for r in evaluations if r.get("status") == "completed"]
    expecteds = [float(r["expected_wall_s"]) for r in evaluations if r.get("status") == "completed"]
    cpus = [float(r.get("cpu_s", 0.0)) for r in evaluations if r.get("status") == "completed"]
    driver_wall = t_end - t_start

    if args.parsl_config == "frontier-slurm":
        slots = args.max_workers_per_node * args.nodes_per_block * args.init_blocks
        nodes = args.nodes_per_block * args.init_blocks
    else:
        slots = args.max_workers_per_node
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1") or "1")

    aggregate_actual = sum(actuals)
    aggregate_cpu = sum(cpus)
    effective_parallelism = aggregate_actual / driver_wall if driver_wall > 0 else None
    slot_occupancy = aggregate_actual / (slots * driver_wall) if slots and driver_wall > 0 else None
    cpu_utilization_est = aggregate_cpu / (slots * driver_wall) if slots and driver_wall > 0 else None

    return {
        "created_at": now_iso(),
        "configuration": {
            "parsl_config": args.parsl_config,
            "arm_grid": args.arm_grid,
            "n_candidates_budget": args.n_candidates,
            "target_wall_s_budget": args.target_wall_s,
            "initial_batch": args.initial_batch,
            "max_in_flight": args.max_in_flight,
            "sleep_scale": args.sleep_scale,
            "noise_sigma": args.noise_sigma,
            "seed": args.seed,
            "nodes_estimate": nodes,
            "worker_slots_estimate": slots,
        },
        "parsl_execution": {
            "tasks_scheduled": len(generation_events),
            "tasks_completed": len(evaluations),
            "tasks_failed": len(failures),
            "driver_wall_s": driver_wall,
            "aggregate_task_actual_wall_s": aggregate_actual,
            "aggregate_task_cpu_s": aggregate_cpu,
            "effective_parallelism_from_task_wall": effective_parallelism,
            "worker_slot_occupancy_estimate": slot_occupancy,
            "app_cpu_utilization_estimate": cpu_utilization_est,
            "note": "Slot occupancy uses task wall time, so sleep-heavy simulated tasks can show high occupancy while CPU utilization is intentionally low.",
        },
        "wall_time_statistics": {
            "expected_mean_s": statistics.mean(expecteds) if expecteds else None,
            "expected_p50_s": percentile(expecteds, 50),
            "expected_p90_s": percentile(expecteds, 90),
            "actual_mean_s": statistics.mean(actuals) if actuals else None,
            "actual_p50_s": percentile(actuals, 50),
            "actual_p90_s": percentile(actuals, 90),
            "actual_min_s": min(actuals) if actuals else None,
            "actual_max_s": max(actuals) if actuals else None,
        },
        "best_result": best,
        "source_counts": {src: sum(1 for r in evaluations if r.get("source") == src) for src in sorted({r.get("source") for r in evaluations})},
        "output_files": {
            "evaluations_csv": "evaluations.csv",
            "evaluations_json": "evaluations.json",
            "best_result_json": "best_result.json",
            "generation_events_csv": "generation_events.csv",
            "strategy_trace_csv": "strategy_trace.csv",
        },
    }


def run_campaign(args: argparse.Namespace) -> Dict[str, Any]:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    arm_stats = {i: {"count": 0, "mean_reward": 0.0, "sum_reward": 0.0} for i in range(args.arm_grid * args.arm_grid)}

    evaluations: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    generation_events: List[Dict[str, Any]] = []
    strategy_trace: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    in_flight: List[Dict[str, Any]] = []
    next_candidate_id = 0

    config = make_config(args)
    parsl.load(config)

    t_start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t_start

    def submit_candidate(candidate: Dict[str, Any], reason: str) -> None:
        nonlocal in_flight
        submit_t = elapsed()
        event = {
            "event": "submitted",
            "reason": reason,
            "time_s": submit_t,
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "x": candidate["x"],
            "y": candidate["y"],
            "arm_id": candidate["arm_id"],
            "grid_arm": candidate["grid_arm"],
            "expected_wall_s": candidate["expected_wall_s"],
            "priority": candidate["priority"],
            "priority_score": candidate["priority_score"],
        }
        generation_events.append(event)
        resource_spec = {"priority": float(candidate["priority"])} if args.use_priority else {}
        try:
            fut = evaluate_candidate(candidate, parsl_resource_specification=resource_spec)
        except TypeError:
            # Older Parsl versions may not accept parsl_resource_specification on app call.
            event["priority_note"] = "parsl_resource_specification rejected; submitted without priority"
            fut = evaluate_candidate(candidate)
        in_flight.append({"future": fut, "candidate": candidate, "submitted_at_s": submit_t})

    try:
        initial = initial_coverage_candidates(args, rng, next_candidate_id)
        for cand in initial[: args.max_in_flight]:
            submit_candidate(cand, reason="initial_batch")
            next_candidate_id += 1

        # If initial_batch > max_in_flight, the extra initial candidates are skipped deliberately;
        # autonomous spawning maintains max_in_flight after the first completions.
        while in_flight:
            done, _ = wait([item["future"] for item in in_flight], timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                # Periodic heartbeat for logs without flooding stdout.
                if int(elapsed()) % max(10, args.heartbeat_s) == 0:
                    pass
                continue

            for fut in list(done):
                idx = next(i for i, item in enumerate(in_flight) if item["future"] is fut)
                item = in_flight.pop(idx)
                candidate = item["candidate"]
                complete_t = elapsed()
                try:
                    result = fut.result()
                    result["submitted_at_s"] = item["submitted_at_s"]
                    result["completed_at_s"] = complete_t
                    result["turnaround_wall_s"] = complete_t - item["submitted_at_s"]
                    result["manager_overhead_estimate_s"] = result["turnaround_wall_s"] - float(result["actual_wall_s"])
                    evaluations.append(result)
                    update_arm_stats(arm_stats, result)
                    if best is None or float(result["objective"]) < float(best["objective"]):
                        best = result

                    trace = {
                        "event": "completed_update",
                        "time_s": complete_t,
                        "candidate_id": result["candidate_id"],
                        "source": result["source"],
                        "arm_id": result["arm_id"],
                        "grid_arm": result["grid_arm"],
                        "objective": result["objective"],
                        "reward": result["reward"],
                        "best_candidate_id": best["candidate_id"] if best else None,
                        "best_objective": best["objective"] if best else None,
                        "arm_count_after": arm_stats[int(result["arm_id"])]["count"],
                        "arm_mean_reward_after": arm_stats[int(result["arm_id"])] ["mean_reward"],
                        "in_flight_after_completion": len(in_flight),
                    }
                    strategy_trace.append(trace)
                except Exception as exc:
                    failure = {
                        "candidate_id": candidate["candidate_id"],
                        "source": candidate["source"],
                        "arm_id": candidate["arm_id"],
                        "grid_arm": candidate["grid_arm"],
                        "submitted_at_s": item["submitted_at_s"],
                        "failed_at_s": complete_t,
                        "error": repr(exc),
                        "status": "failed",
                    }
                    failures.append(failure)
                    strategy_trace.append({"event": "failed", **failure})

                # Autonomous mode: one completed result triggers one new candidate if budgets allow.
                can_spawn = (
                    args.autonomous
                    and next_candidate_id < args.n_candidates
                    and elapsed() < args.target_wall_s
                    and len(in_flight) < args.max_in_flight
                )
                if can_spawn:
                    source = choose_source(args, rng, best)
                    new_cand = make_candidate(args, rng, next_candidate_id, source, best=best, arm_stats=arm_stats)
                    # Save extra strategy info before submission.
                    strategy_trace.append(
                        {
                            "event": "spawn_decision",
                            "time_s": elapsed(),
                            "candidate_id": new_cand["candidate_id"],
                            "source": new_cand["source"],
                            "arm_id": new_cand["arm_id"],
                            "grid_arm": new_cand["grid_arm"],
                            "expected_wall_s": new_cand["expected_wall_s"],
                            "priority": new_cand["priority"],
                            "priority_score": new_cand["priority_score"],
                            "best_candidate_id": best["candidate_id"] if best else None,
                            "best_objective": best["objective"] if best else None,
                            "thompson_trace": new_cand.get("thompson_trace", {}),
                        }
                    )
                    submit_candidate(new_cand, reason="autonomous_after_completion")
                    next_candidate_id += 1

            # Stop spawning after wall budget, but allow already scheduled work to finish.
            if elapsed() >= args.target_wall_s and args.stop_after_target:
                args.autonomous = False

    finally:
        t_end = time.perf_counter()
        try:
            parsl.clear()
        except Exception:
            pass

    # Sort by candidate_id for deterministic files.
    evaluations.sort(key=lambda r: int(r["candidate_id"]))
    generation_events.sort(key=lambda r: int(r["candidate_id"]))

    summary = build_summary(args, evaluations, failures, generation_events, strategy_trace, best, t_start, t_end)
    monitoring_summary = summarize_monitoring_db([Path.cwd(), Path(args.run_dir).resolve(), outdir])
    summary["parsl_monitoring"] = monitoring_summary

    write_csv(outdir / "evaluations.csv", evaluations)
    write_json(outdir / "evaluations.json", evaluations)
    write_json(outdir / "best_result.json", best or {})
    write_csv(outdir / "generation_events.csv", generation_events)
    write_json(outdir / "generation_events.json", generation_events)
    write_csv(outdir / "strategy_trace.csv", strategy_trace)
    write_json(outdir / "strategy_trace.json", strategy_trace)
    write_json(outdir / "failures.json", failures)
    write_json(outdir / "summary.json", summary)
    write_json(outdir / "parsl_monitoring_summary.json", monitoring_summary)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive asynchronous Parsl Rastrigin optimization campaign")
    p.add_argument("--parsl-config", choices=["local-htex", "frontier-slurm"], default="local-htex")
    p.add_argument("--outdir", default="rastrigin_parsl_outputs")
    p.add_argument("--run-dir", default="parsl_runinfo")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--arm-grid", type=int, default=8)
    p.add_argument("--n-candidates", type=int, default=128)
    p.add_argument("--target-wall-s", type=float, default=240.0)
    p.add_argument("--initial-batch", type=int, default=32)
    p.add_argument("--max-in-flight", type=int, default=56)
    p.add_argument("--autonomous", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-after-target", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--p-random", type=float, default=0.25)
    p.add_argument("--p-local", type=float, default=0.35)
    p.add_argument("--p-thompson", type=float, default=0.40)
    p.add_argument("--prior-mean-reward", type=float, default=-30.0)
    p.add_argument("--prior-reward-scale", type=float, default=30.0)
    p.add_argument("--priority-reward-shift", type=float, default=80.0)
    p.add_argument("--use-priority", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--noise-sigma", type=float, default=0.10)
    p.add_argument("--sleep-scale", type=float, default=1.0)
    p.add_argument("--min-wall-s", type=float, default=1.0)
    p.add_argument("--max-wall-s", type=float, default=20.0)
    p.add_argument("--straggler-probability", type=float, default=0.10)

    # Resource / Parsl options.
    p.add_argument("--cores-per-worker", type=float, default=1.0)
    p.add_argument("--max-workers-per-node", type=int, default=56)
    p.add_argument("--account", default=os.environ.get("OLCF_ACCOUNT", ""), help="OLCF project ID for Frontier SlurmProvider mode")
    p.add_argument("--partition", default="batch")
    p.add_argument("--nodes-per-block", type=int, default=1)
    p.add_argument("--init-blocks", type=int, default=1)
    p.add_argument("--max-blocks", type=int, default=1)
    p.add_argument("--block-walltime", default="00:10:00")
    p.add_argument("--job-name", default="parsl-rastrigin")
    p.add_argument("--core-spec", type=int, default=8, help="Frontier Slurm core specialization value; default 8 leaves 56 allocatable cores")
    p.add_argument("--frontier-interface", default="hsn0")
    p.add_argument("--worker-init", default="")
    p.add_argument("--extra-sbatch", action="append", default=[], help="Extra #SBATCH line, e.g. '#SBATCH --constraint=...' ; can be repeated")

    p.add_argument("--monitoring", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--monitoring-port", type=int, default=55055)
    p.add_argument("--resource-monitoring-interval", type=int, default=10)
    p.add_argument("--heartbeat-s", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.arm_grid < 1:
        raise SystemExit("--arm-grid must be >= 1")
    if args.initial_batch < 1 or args.max_in_flight < 1:
        raise SystemExit("--initial-batch and --max-in-flight must be >= 1")
    if args.p_random + args.p_local + args.p_thompson <= 0:
        raise SystemExit("At least one source probability must be positive")

    summary = run_campaign(args)
    print(json.dumps({
        "outdir": str(Path(args.outdir).resolve()),
        "tasks_scheduled": summary["parsl_execution"]["tasks_scheduled"],
        "tasks_completed": summary["parsl_execution"]["tasks_completed"],
        "driver_wall_s": summary["parsl_execution"]["driver_wall_s"],
        "best_objective": summary["best_result"].get("objective") if summary.get("best_result") else None,
        "best_x": summary["best_result"].get("x") if summary.get("best_result") else None,
        "best_y": summary["best_result"].get("y") if summary.get("best_result") else None,
        "worker_slot_occupancy_estimate": summary["parsl_execution"].get("worker_slot_occupancy_estimate"),
        "app_cpu_utilization_estimate": summary["parsl_execution"].get("app_cpu_utilization_estimate"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
