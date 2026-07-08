"""
Asynchronous Thompson-sampling demo on a multimodal Rastrigin objective.

This is a MatEnsemble scheduling/strategy showcase rather than a materials
physics run. It uses many short, heterogeneous CPU chores with deliberately
wide wall times. A custom FutureProcessingStrategy updates a posterior as
results arrive and reorders the remaining ready queue before backfilling freed
resources.

API note: the GitHub MatEnsemble strategy example uses a newer
``@pipe.strategy`` / BOLO-list pattern to spawn new chores from completed
results. The Frontier sandbox inspected for this INCITE work
(`matensemble-docker-steps/Test-Sadnbox`) exposes the older/lower-level
``FutureProcessingStrategy`` interface instead. This workflow therefore uses
that local interface directly so it can reorder an already populated ready
queue and demonstrate adaptive scheduling under heterogeneous wall times.
"""

import concurrent.futures
import csv
import json
import math
import os
import random
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

from matensemble.manager import FluxManager
from matensemble.pipeline import Pipeline
from matensemble.strategy import FutureProcessingStrategy, append_text

N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "768"))
ARM_GRID = int(os.environ.get("ARM_GRID", "8"))
SEED = int(os.environ.get("SEED", "20260613"))
MIN_SLEEP = float(os.environ.get("MIN_SLEEP", "0.15"))
MAX_SLEEP = float(os.environ.get("MAX_SLEEP", "12.0"))
SLEEP_SCALE = float(os.environ.get("SLEEP_SCALE", "1.2"))
NOISE = float(os.environ.get("NOISE", "0.05"))
COST_POWER = float(os.environ.get("COST_POWER", "0.35"))
AUTONOMOUS_MODE = os.environ.get("AUTONOMOUS_MODE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
TARGET_WALL_S = float(os.environ.get("TARGET_WALL_S", "0.0"))
INITIAL_TASKS = int(os.environ.get("INITIAL_TASKS", "224"))
LOG_DELAY = float(os.environ.get("LOG_DELAY", "5.0"))


def rastrigin(x, y):
    return (
        20.0
        + x * x
        - 10.0 * math.cos(2.0 * math.pi * x)
        + y * y
        - 10.0 * math.cos(2.0 * math.pi * y)
    )


def arm_for_xy(x, y, arm_grid):
    scale = arm_grid / 10.24
    ix = max(0, min(arm_grid - 1, int((x + 5.12) * scale)))
    iy = max(0, min(arm_grid - 1, int((y + 5.12) * scale)))
    return iy * arm_grid + ix


def wall_time_for_candidate(x, y, candidate_id, seed):
    wall_rng = random.Random(seed * 1000003 + int(candidate_id))
    raw = wall_rng.lognormvariate(0.0, 1.05)
    spatial = 1.0 + 0.65 * abs(math.sin(1.7 * x) * math.cos(1.3 * y))
    return max(MIN_SLEEP, min(MAX_SLEEP, MIN_SLEEP + SLEEP_SCALE * raw * spatial))


def candidate_for_xy(candidate_id, x, y, arm_grid, seed):
    return {
        "candidate_id": int(candidate_id),
        "x": float(x),
        "y": float(y),
        "arm": arm_for_xy(x, y, arm_grid),
        "expected_wall_s": wall_time_for_candidate(x, y, candidate_id, seed),
    }


def candidate_in_arm(candidate_id, arm, arm_grid, rng, seed):
    ix = int(arm) % arm_grid
    iy = int(arm) // arm_grid
    dx = 10.24 / arm_grid
    x0 = -5.12 + ix * dx
    y0 = -5.12 + iy * dx
    x = rng.uniform(x0, x0 + dx)
    y = rng.uniform(y0, y0 + dx)
    return candidate_for_xy(candidate_id, x, y, arm_grid, seed)


def build_candidates(n_candidates, arm_grid, seed):
    rng = random.Random(seed)
    candidates = []
    for cid in range(n_candidates):
        if cid < arm_grid * arm_grid:
            ix = cid % arm_grid
            iy = cid // arm_grid
            x = -5.12 + (ix + 0.5) * 10.24 / arm_grid
            y = -5.12 + (iy + 0.5) * 10.24 / arm_grid
        else:
            # Mix global coverage with extra density near the true basin.
            if rng.random() < 0.32:
                x = max(-5.12, min(5.12, rng.gauss(0.0, 1.35)))
                y = max(-5.12, min(5.12, rng.gauss(0.0, 1.35)))
            else:
                x = rng.uniform(-5.12, 5.12)
                y = rng.uniform(-5.12, 5.12)

        candidates.append(candidate_for_xy(cid, x, y, arm_grid, seed))

    # Initial order: broad one-per-arm exploration, then shuffled remainder.
    first = candidates[: arm_grid * arm_grid]
    rest = candidates[arm_grid * arm_grid :]
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


pipe = Pipeline()


@pipe.chore(
    name="rastrigin-eval",
    num_tasks=1,
    cores_per_task=1,
    gpus_per_task=0,
    mpi=False,
    inherit_env=True,
)
def evaluate_candidate(candidate, noise, seed):
    import math as _math
    import random as _random
    import time as _time

    rng = _random.Random(seed + int(candidate["candidate_id"]) * 7919)
    t0 = _time.perf_counter()
    # Wide wall-time distribution plus small deterministic jitter.
    sleep_s = candidate["expected_wall_s"] * rng.uniform(0.75, 1.35)
    _time.sleep(sleep_s)

    x = float(candidate["x"])
    y = float(candidate["y"])
    objective = (
        20.0
        + x * x
        - 10.0 * _math.cos(2.0 * _math.pi * x)
        + y * y
        - 10.0 * _math.cos(2.0 * _math.pi * y)
    )
    objective += rng.gauss(0.0, noise)
    reward = -objective
    elapsed = _time.perf_counter() - t0
    return {
        "candidate_id": int(candidate["candidate_id"]),
        "x": x,
        "y": y,
        "arm": int(candidate["arm"]),
        "source": str(candidate.get("source", "prebuilt")),
        "expected_wall_s": float(candidate["expected_wall_s"]),
        "actual_wall_s": elapsed,
        "objective": objective,
        "reward": reward,
    }


class ThompsonReorderStrategy(FutureProcessingStrategy):
    def __init__(
        self,
        manager,
        arm_grid,
        seed,
        cost_power,
        pipeline=None,
        autonomous=False,
        max_evals=0,
        target_wall_s=0.0,
        initial_count=0,
        noise=0.05,
    ):
        self.manager = manager
        self.pipeline = pipeline
        self.arm_grid = arm_grid
        self.seed = seed
        self.rng = random.Random(seed + 999)
        self.cost_power = cost_power
        self.autonomous = autonomous
        self.max_evals = max_evals
        self.target_wall_s = target_wall_s
        self.next_candidate_id = initial_count
        self.noise = noise
        self.start = time.perf_counter()
        self.trace_path = manager._base_dir / "strategy_trace.jsonl"
        self.arm_stats = {
            arm: {"n": 0, "mean": -42.0, "m2": 0.0}
            for arm in range(arm_grid * arm_grid)
        }
        self.best_reward = None
        self.best_result = None

    def _now(self):
        return time.perf_counter() - self.start

    def _write_event(self, event):
        event["time_s"] = self._now()
        with open(self.trace_path, "a") as fh:
            fh.write(json.dumps(event) + "\n")

    def _candidate_from_chore(self, chore_id):
        chore = self.manager._chores_by_id[chore_id]
        if chore.args:
            return chore.args[0]
        return chore.kwargs.get("candidate", {})

    def _campaign_open_for_generation(self):
        if not self.autonomous:
            return False
        if self.pipeline is None:
            return False
        if self.next_candidate_id >= self.max_evals:
            return False
        if self.target_wall_s > 0.0 and self._now() >= self.target_wall_s:
            return False
        return True

    def _score_ready_chore(self, chore_id):
        candidate = self._candidate_from_chore(chore_id)
        arm = int(candidate.get("arm", 0))
        stats = self.arm_stats.get(arm, {"n": 0, "mean": -42.0, "m2": 0.0})
        n = stats["n"]
        # Conservative normal posterior over reward; high reward is better.
        sigma = 25.0 / math.sqrt(n + 1.0)
        reward_sample = self.rng.gauss(stats["mean"], sigma)
        cost = max(0.05, float(candidate.get("expected_wall_s", 1.0)))
        return reward_sample / (cost**self.cost_power)

    def _sample_arm(self):
        scored = []
        for arm in range(self.arm_grid * self.arm_grid):
            stats = self.arm_stats[arm]
            n = stats["n"]
            sigma = 25.0 / math.sqrt(n + 1.0)
            reward_sample = self.rng.gauss(stats["mean"], sigma)
            scored.append((reward_sample, arm))
        scored.sort(reverse=True)
        return scored[0][1]

    def _generate_candidate(self):
        cid = self.next_candidate_id
        self.next_candidate_id += 1

        # Keep a little global exploration. Otherwise draw an arm from the
        # Thompson posterior and sample within that arm.
        if self.rng.random() < 0.10:
            x = self.rng.uniform(-5.12, 5.12)
            y = self.rng.uniform(-5.12, 5.12)
            candidate = candidate_for_xy(cid, x, y, self.arm_grid, self.seed)
            source = "global"
        elif self.best_result and self.rng.random() < 0.22:
            x = max(
                -5.12, min(5.12, self.rng.gauss(float(self.best_result["x"]), 0.55))
            )
            y = max(
                -5.12, min(5.12, self.rng.gauss(float(self.best_result["y"]), 0.55))
            )
            candidate = candidate_for_xy(cid, x, y, self.arm_grid, self.seed)
            source = "local-best"
        else:
            arm = self._sample_arm()
            candidate = candidate_in_arm(cid, arm, self.arm_grid, self.rng, self.seed)
            source = "thompson-arm"

        candidate["source"] = source
        return candidate

    def _register_chore(self, chore):
        self.manager._chores_by_id[chore.id] = chore
        self.manager._dependents[chore.id] = []
        self.manager._remaining_deps[chore.id] = len(chore.deps)
        if self.manager._remaining_deps[chore.id] == 0:
            self.manager._ready.append(chore.id)
        else:
            self.manager._blocked.add(chore.id)

    def _add_autonomous_candidate(self):
        if not self._campaign_open_for_generation():
            return None
        candidate = self._generate_candidate()
        evaluate_candidate(candidate, self.noise, self.seed)
        chore = self.pipeline._chore_list[-1]
        self._register_chore(chore)
        self._write_event(
            {
                "event": "generate",
                "chore_id": chore.id,
                "candidate_id": candidate["candidate_id"],
                "arm": candidate["arm"],
                "source": candidate.get("source", ""),
                "expected_wall_s": candidate["expected_wall_s"],
                "generated_total": self.next_candidate_id,
                "target_wall_s": self.target_wall_s,
            }
        )
        return chore.id

    def _reorder_ready(self):
        ready = list(self.manager._ready)
        if not ready:
            return
        scored = [(self._score_ready_chore(chore_id), chore_id) for chore_id in ready]
        scored.sort(reverse=True)
        self.manager._ready = deque([chore_id for _, chore_id in scored])
        self._write_event(
            {
                "event": "reorder",
                "ready": len(ready),
                "running": len(self.manager._running_chores),
                "top_ready": [chore_id for _, chore_id in scored[:8]],
            }
        )

    def _update_posterior(self, result):
        arm = int(result["arm"])
        reward = float(result["reward"])
        stats = self.arm_stats[arm]
        stats["n"] += 1
        delta = reward - stats["mean"]
        stats["mean"] += delta / stats["n"]
        stats["m2"] += delta * (reward - stats["mean"])
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
            self.best_result = result

    def _read_result_json(self, chore):
        path = chore.workdir / "result.json"
        if not path.exists():
            return None
        with open(path) as fh:
            return json.load(fh)

    def process_futures(self, buffer_time):
        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )

        for fut in completed:
            chore_id = fut.chore_id
            chore = fut.chore_obj
            self.manager._running_chores.remove(chore_id)

            try:
                rc = fut.result()
            except Exception as exc:
                tb = traceback.format_exc()
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_text(
                    chore.workdir / "stderr",
                    "\n\n===== TS STRATEGY WRAPPER ERROR (%s) =====\nchore=%s\n%s: %s\n%s\n"
                    % (stamp, chore_id, type(exc).__name__, exc, tb),
                )
                self.manager._record_failure(
                    chore_id,
                    reason="exception",
                    exception="%s: %s" % (type(exc).__name__, exc),
                )
                self.manager._fail_dependents(chore_id)
                continue

            if rc != 0:
                append_text(
                    chore.workdir / "stderr",
                    "\n\n===== MATENSEMBLE: NONZERO EXIT =====\nchore=%s rc=%s\n"
                    % (chore_id, rc),
                )
                self.manager._record_failure(chore_id, reason="nonzero_exit:%s" % rc)
                self.manager._fail_dependents(chore_id)
                continue

            self.manager._completed_chores.append(chore_id)
            result = self._read_result_json(chore)
            if result:
                self._update_posterior(result)
                self._write_event(
                    {
                        "event": "complete",
                        "chore_id": chore_id,
                        "candidate_id": result["candidate_id"],
                        "arm": result["arm"],
                        "objective": result["objective"],
                        "reward": result["reward"],
                        "actual_wall_s": result["actual_wall_s"],
                        "best_reward": self.best_reward,
                        "best_candidate_id": self.best_result["candidate_id"],
                        "pending": len(self.manager._ready)
                        + len(self.manager._blocked),
                        "running": len(self.manager._running_chores),
                        "completed": len(self.manager._completed_chores),
                    }
                )

            for dep_id in self.manager._dependents.get(chore_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            self._add_autonomous_candidate()
            self._reorder_ready()
            self.manager._submit_until_ooresources(buffer_time=buffer_time)


def build_manager_for_pipeline(pipeline):
    dag = pipeline._create_graph()
    ordered_ids = pipeline._sort_graph(dag)
    ordered_chores = [dag.nodes[chore_id]["chore"] for chore_id in ordered_ids]
    pipeline._out_dir.mkdir(parents=True, exist_ok=True)
    return FluxManager(
        chore_list=ordered_chores,
        base_dir=pipeline._base_dir,
        write_restart_freq=None,
        set_cpu_affinity=True,
        set_gpu_affinity=False,
    )


def collect_results(workflow_dir):
    rows = []
    for path in sorted(Path(workflow_dir).glob("out/chore-*/result.json")):
        with open(path) as fh:
            rows.append(json.load(fh))
    rows.sort(key=lambda row: row["candidate_id"])
    return rows


def write_summary(workflow_dir, rows, trace_path):
    completion_events = []
    generation_events = []
    reorder_events = 0
    if trace_path.exists():
        with open(trace_path) as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("event") == "complete":
                    completion_events.append(event)
                elif event.get("event") == "generate":
                    generation_events.append(event)
                elif event.get("event") == "reorder":
                    reorder_events += 1

    best = min(rows, key=lambda row: row["objective"]) if rows else None
    summary = {
        "n_candidates": len(rows),
        "arm_grid": ARM_GRID,
        "seed": SEED,
        "autonomous_mode": AUTONOMOUS_MODE,
        "target_wall_s": TARGET_WALL_S,
        "initial_tasks": INITIAL_TASKS,
        "configured_max_evals": N_CANDIDATES,
        "min_sleep_s": MIN_SLEEP,
        "max_sleep_s": MAX_SLEEP,
        "sleep_scale": SLEEP_SCALE,
        "best": best,
        "mean_wall_s": (
            sum(row["actual_wall_s"] for row in rows) / len(rows) if rows else 0.0
        ),
        "max_wall_s": max([row["actual_wall_s"] for row in rows] or [0.0]),
        "completion_events": completion_events,
        "generation_events": generation_events,
        "n_generation_events": len(generation_events),
        "n_reorder_events": reorder_events,
    }
    with open("adaptive_ts_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open("adaptive_ts_results.csv", "w", newline="") as fh:
        fieldnames = [
            "candidate_id",
            "x",
            "y",
            "arm",
            "source",
            "expected_wall_s",
            "actual_wall_s",
            "objective",
            "reward",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("\n=== ADAPTIVE THOMPSON SAMPLING SUMMARY ===")
    print("evaluations: %d" % len(rows))
    if best:
        print(
            "best candidate=%d x=(%.3f, %.3f) objective=%.4f wall=%.2fs"
            % (
                best["candidate_id"],
                best["x"],
                best["y"],
                best["objective"],
                best["actual_wall_s"],
            )
        )
    print("results: adaptive_ts_results.json adaptive_ts_results.csv")


initial_count = min(N_CANDIDATES, INITIAL_TASKS) if AUTONOMOUS_MODE else N_CANDIDATES
print(
    "[ATS] Building %d initial candidates on %dx%d Rastrigin arms "
    "(autonomous=%s max_evals=%d target_wall_s=%.1f)"
    % (initial_count, ARM_GRID, ARM_GRID, AUTONOMOUS_MODE, N_CANDIDATES, TARGET_WALL_S),
    flush=True,
)
candidates = build_candidates(initial_count, ARM_GRID, SEED)
for candidate in candidates:
    evaluate_candidate(candidate, NOISE, SEED)

manager = build_manager_for_pipeline(pipe)
strategy = ThompsonReorderStrategy(
    manager,
    ARM_GRID,
    SEED,
    COST_POWER,
    pipeline=pipe,
    autonomous=AUTONOMOUS_MODE,
    max_evals=N_CANDIDATES,
    target_wall_s=TARGET_WALL_S,
    initial_count=len(candidates),
    noise=NOISE,
)
t0 = time.perf_counter()
manager.run(
    buffer_time=0.0,
    log_delay=LOG_DELAY,
    adaptive=True,
    processing_strategy=strategy,
)
elapsed = time.perf_counter() - t0

workflow_dir = manager._base_dir
rows = collect_results(workflow_dir)
write_summary(workflow_dir, rows, strategy.trace_path)
print("workflow elapsed: %.2f s" % elapsed)

status_path = Path(workflow_dir) / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}
if status.get("failed", 0):
    raise SystemExit(1)
