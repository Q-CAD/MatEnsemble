"""
Low-level adaptive Thompson-sampling demo on a multimodal Rastrigin objective.

This example intentionally avoids Pipeline. It registers Python chores with a
ChoreRegistry, builds Chore objects directly, creates its own FluxManager, and
uses a custom FutureProcessingStrategy to update the posterior, reorder the
ready queue, and spawn additional evaluation chores.
"""

import concurrent.futures
import copy
import csv
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

from matensemble.chore import Chore, ChoreRegistry
from matensemble.manager import FluxManager
from matensemble.model import ChoreType, Resources
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

registry = ChoreRegistry()


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
    return candidate_for_xy(
        candidate_id,
        rng.uniform(x0, x0 + dx),
        rng.uniform(y0, y0 + dx),
        arm_grid,
        seed,
    )


def build_candidates(n_candidates, arm_grid, seed):
    rng = random.Random(seed)
    candidates = []
    for cid in range(n_candidates):
        if cid < arm_grid * arm_grid:
            ix = cid % arm_grid
            iy = cid // arm_grid
            x = -5.12 + (ix + 0.5) * 10.24 / arm_grid
            y = -5.12 + (iy + 0.5) * 10.24 / arm_grid
        elif rng.random() < 0.32:
            x = max(-5.12, min(5.12, rng.gauss(0.0, 1.35)))
            y = max(-5.12, min(5.12, rng.gauss(0.0, 1.35)))
        else:
            x = rng.uniform(-5.12, 5.12)
            y = rng.uniform(-5.12, 5.12)
        candidates.append(candidate_for_xy(cid, x, y, arm_grid, seed))

    first = candidates[: arm_grid * arm_grid]
    rest = candidates[arm_grid * arm_grid :]
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


@registry.chore(
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
    _time.sleep(candidate["expected_wall_s"] * rng.uniform(0.75, 1.35))

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
        "reward": -objective,
    }


class ThompsonLowLevelStrategy(FutureProcessingStrategy):
    def __init__(
        self,
        manager,
        out_dir,
        arm_grid,
        seed,
        cost_power,
        next_candidate_id,
        autonomous,
        max_evals,
        target_wall_s,
        noise,
    ):
        super().__init__(manager)
        self.out_dir = Path(out_dir)
        self.arm_grid = arm_grid
        self.seed = seed
        self.rng = random.Random(seed + 999)
        self.cost_power = cost_power
        self.next_candidate_id = next_candidate_id
        self.autonomous = autonomous
        self.max_evals = max_evals
        self.target_wall_s = target_wall_s
        self.noise = noise
        self.start = time.perf_counter()
        self.trace_path = manager._base_dir / "strategy_trace_low_level.jsonl"
        self.arm_stats = {
            arm: {"n": 0, "mean": -42.0, "m2": 0.0}
            for arm in range(arm_grid * arm_grid)
        }
        self.best_reward = None
        self.best_result = None
        self.chore_counter = len(manager._chores_by_id)

    def _write_event(self, event):
        event["time_s"] = time.perf_counter() - self.start
        with self.trace_path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")

    def _candidate_from_chore(self, chore_id):
        chore = self.manager._chores_by_id[chore_id]
        return chore.args[0] if chore.args else chore.kwargs.get("candidate", {})

    def _score_candidate(self, candidate):
        arm = int(candidate.get("arm", 0))
        stats = self.arm_stats[arm]
        sigma = 25.0 / math.sqrt(stats["n"] + 1.0)
        reward_sample = self.rng.gauss(stats["mean"], sigma)
        cost = max(0.05, float(candidate.get("expected_wall_s", 1.0)))
        return reward_sample / (cost**self.cost_power)

    def _nice_for_candidate(self, candidate):
        return max(-1000, min(1000, int(-10.0 * self._score_candidate(candidate))))

    def _make_chore(self, candidate):
        entry = registry.get("rastrigin-eval")
        resources = copy.deepcopy(entry.resources)
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        source_bits = [str(Path.cwd()), str((Path.cwd() / "src").resolve())]
        resources.env = {"PYTHONPATH": ":".join(source_bits + [old_pythonpath])}
        self.chore_counter += 1
        chore_id = f"chore-{entry.id_name}-{self.chore_counter:04d}"
        workdir = self.out_dir / chore_id
        return Chore(
            id=chore_id,
            workdir=workdir,
            command=[
                sys.executable,
                "-m",
                "matensemble.runtime_worker",
                "--chore-id",
                chore_id,
                "--spec-file",
                str(workdir / "chore.pickle"),
            ],
            chore_type=ChoreType.PYTHON,
            resources=resources,
            chore_qualname=entry.qualname,
            args=(copy.deepcopy(candidate), self.noise, self.seed),
            kwargs={},
            nice=self._nice_for_candidate(candidate),
        )

    def _generate_candidate(self):
        cid = self.next_candidate_id
        self.next_candidate_id += 1
        if self.rng.random() < 0.10:
            candidate = candidate_for_xy(
                cid,
                self.rng.uniform(-5.12, 5.12),
                self.rng.uniform(-5.12, 5.12),
                self.arm_grid,
                self.seed,
            )
            source = "global"
        elif self.best_result and self.rng.random() < 0.22:
            candidate = candidate_for_xy(
                cid,
                max(-5.12, min(5.12, self.rng.gauss(self.best_result["x"], 0.55))),
                max(-5.12, min(5.12, self.rng.gauss(self.best_result["y"], 0.55))),
                self.arm_grid,
                self.seed,
            )
            source = "local-best"
        else:
            scored = []
            for arm in range(self.arm_grid * self.arm_grid):
                stats = self.arm_stats[arm]
                sigma = 25.0 / math.sqrt(stats["n"] + 1.0)
                scored.append((self.rng.gauss(stats["mean"], sigma), arm))
            scored.sort(reverse=True)
            candidate = candidate_in_arm(
                cid, scored[0][1], self.arm_grid, self.rng, self.seed
            )
            source = "thompson-arm"
        candidate["source"] = source
        return candidate

    def _campaign_open(self):
        if not self.autonomous or self.next_candidate_id >= self.max_evals:
            return False
        return (
            self.target_wall_s <= 0.0
            or time.perf_counter() - self.start < self.target_wall_s
        )

    def _reorder_ready(self):
        scored = []
        for chore_id in self.manager._ready:
            chore = self.manager._chores_by_id[chore_id]
            candidate = self._candidate_from_chore(chore_id)
            score = self._score_candidate(candidate)
            chore.nice = max(-1000, min(1000, int(-10.0 * score)))
            scored.append((score, chore_id))
        scored.sort(reverse=True)
        self.manager._ready = deque([chore_id for _, chore_id in scored])
        if scored:
            self._write_event(
                {
                    "event": "reorder",
                    "ready": len(scored),
                    "running": len(self.manager._running_chores),
                    "top_ready": [chore_id for _, chore_id in scored[:8]],
                }
            )

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

            if rc != 0 and rc != 134:
                append_text(
                    chore.workdir / "stderr",
                    "\n\n===== MATENSEMBLE: NONZERO EXIT =====\nchore=%s rc=%s\n"
                    % (chore_id, rc),
                )
                self.manager._record_failure(chore_id, reason="nonzero_exit:%s" % rc)
                self.manager._fail_dependents(chore_id)
                continue

            self.manager._completed_chores.append(chore_id)
            result_path = chore.workdir / "result.pickle"
            result = (
                pickle.loads(result_path.read_bytes()) if result_path.exists() else None
            )
            if result:
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
                        "completed": len(self.manager._completed_chores),
                    }
                )

            for dep_id in self.manager._dependents.get(chore_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._mark_ready(dep_id)
                    self.manager._blocked.discard(dep_id)

            if self._campaign_open():
                candidate = self._generate_candidate()
                new_chore = self._make_chore(candidate)
                self.manager._add_chore(new_chore)
                self._write_event(
                    {
                        "event": "generate",
                        "chore_id": new_chore.id,
                        "candidate_id": candidate["candidate_id"],
                        "arm": candidate["arm"],
                        "source": candidate["source"],
                        "expected_wall_s": candidate["expected_wall_s"],
                        "nice": new_chore.nice,
                        "generated_total": self.next_candidate_id,
                    }
                )

            self._reorder_ready()
            self.manager._submit_until_ooresources(buffer_time=buffer_time)


def write_summary(workflow_dir, rows, trace_path):
    completion_events = []
    generation_events = []
    reorder_events = 0
    if trace_path.exists():
        with trace_path.open() as fh:
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
        "workflow_api": "ChoreRegistry + FluxManager + FutureProcessingStrategy",
        "n_candidates": len(rows),
        "arm_grid": ARM_GRID,
        "seed": SEED,
        "autonomous_mode": AUTONOMOUS_MODE,
        "target_wall_s": TARGET_WALL_S,
        "initial_tasks": INITIAL_TASKS,
        "configured_max_evals": N_CANDIDATES,
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
    with open("adaptive_ts_results_low_level.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open("adaptive_ts_results_low_level.csv", "w", newline="") as fh:
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

    print("\n=== ADAPTIVE THOMPSON SAMPLING SUMMARY (LOW-LEVEL API) ===")
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
    print(
        "results: adaptive_ts_results_low_level.json adaptive_ts_results_low_level.csv"
    )


initial_count = min(N_CANDIDATES, INITIAL_TASKS) if AUTONOMOUS_MODE else N_CANDIDATES
print(
    "[ATS low-level API] Building %d initial candidates on %dx%d Rastrigin arms "
    "(autonomous=%s max_evals=%d target_wall_s=%.1f)"
    % (initial_count, ARM_GRID, ARM_GRID, AUTONOMOUS_MODE, N_CANDIDATES, TARGET_WALL_S),
    flush=True,
)

base_dir = Path.cwd() / f"matensemble_workflow-low-level-{datetime.now():%Y%m%d_%H%M%S}"
out_dir = base_dir / "out"
registry.write(out_dir / "registry")

initial_candidates = build_candidates(initial_count, ARM_GRID, SEED)
chores = []
old_pythonpath = os.environ.get("PYTHONPATH", "")
source_bits = [str(Path.cwd()), str((Path.cwd() / "src").resolve())]
for idx, candidate in enumerate(initial_candidates, start=1):
    entry = registry.get("rastrigin-eval")
    resources = copy.deepcopy(entry.resources)
    resources.env = {"PYTHONPATH": ":".join(source_bits + [old_pythonpath])}
    chore_id = f"chore-{entry.id_name}-{idx:04d}"
    workdir = out_dir / chore_id
    nice = max(-1000, min(1000, int(10.0 * candidate["expected_wall_s"])))
    chores.append(
        Chore(
            id=chore_id,
            workdir=workdir,
            command=[
                sys.executable,
                "-m",
                "matensemble.runtime_worker",
                "--chore-id",
                chore_id,
                "--spec-file",
                str(workdir / "chore.pickle"),
            ],
            chore_type=ChoreType.PYTHON,
            resources=resources,
            chore_qualname=entry.qualname,
            args=(copy.deepcopy(candidate), NOISE, SEED),
            kwargs={},
            nice=nice,
        )
    )

manager = FluxManager(
    chore_list=chores,
    base_dir=base_dir,
    write_restart_freq=None,
    set_cpu_affinity=True,
    set_gpu_affinity=False,
)
strategy = ThompsonLowLevelStrategy(
    manager=manager,
    out_dir=out_dir,
    arm_grid=ARM_GRID,
    seed=SEED,
    cost_power=COST_POWER,
    next_candidate_id=len(initial_candidates),
    autonomous=AUTONOMOUS_MODE,
    max_evals=N_CANDIDATES,
    target_wall_s=TARGET_WALL_S,
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

rows = []
for path in sorted(out_dir.glob("chore-rastrigin-eval-*/result.pickle")):
    with path.open("rb") as fh:
        rows.append(pickle.load(fh))
rows.sort(key=lambda row: row["candidate_id"])
write_summary(manager._base_dir, rows, strategy.trace_path)
print("workflow elapsed: %.2f s" % elapsed)

status_path = manager._base_dir / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}
if status.get("failed", 0):
    raise SystemExit(1)
