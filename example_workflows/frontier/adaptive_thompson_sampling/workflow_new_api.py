"""
Thompson-sampling demo using the Pipeline.strategy / ChoreSpec API.

This workflow is intentionally similar to ``workflow.py`` in this directory,
but it uses the newer MatEnsemble strategy API instead of a custom
FutureProcessingStrategy. The tradeoff is important:

* New evaluation chores are spawned by returning ChoreSpec objects.
* Spawned chores are sorted into FluxManager's ready queue by their nice value.
* This version does not manually reorder already-ready chores.

The strategy callback itself is a chore, so persistent Thompson-sampling state
is stored in a small JSON file under the workflow directory.
"""

import csv
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

from matensemble.chore import ChoreSpec
from matensemble.model import Resources
from matensemble.pipeline import Pipeline

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


pipe = Pipeline()
STATE_PATH = pipe._base_dir / "thompson_state.json"
TRACE_PATH = pipe._base_dir / "strategy_trace_new_api.jsonl"


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


@pipe.strategy(
    bolo_list=["rastrigin-eval"],
    name="thompson-next",
    num_tasks=1,
    cores_per_task=1,
    gpus_per_task=0,
    mpi=False,
    inherit_env=True,
)
def thompson_next(result):
    import json as _json
    import math as _math
    import os as _os
    import random as _random
    import time as _time
    from pathlib import Path as _Path

    state_path = _Path(_os.environ["MATENSEMBLE_TS_STATE"])
    trace_path = _Path(_os.environ["MATENSEMBLE_TS_TRACE"])
    lock_path = state_path.with_suffix(".lock")

    def acquire_lock(timeout_s=120.0):
        start = _time.monotonic()
        while True:
            try:
                return _os.open(
                    lock_path,
                    _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if _time.monotonic() - start > timeout_s:
                    raise TimeoutError(f"timed out waiting for {lock_path}")
                _time.sleep(0.05)

    def write_event(event, state):
        event["time_s"] = _time.time() - state["start_epoch"]
        with trace_path.open("a") as fh:
            fh.write(_json.dumps(event) + "\n")

    def sample_arm(state, rng):
        scored = []
        for arm in range(state["arm_grid"] * state["arm_grid"]):
            stats = state["arm_stats"][str(arm)]
            sigma = 25.0 / _math.sqrt(stats["n"] + 1.0)
            scored.append((rng.gauss(stats["mean"], sigma), arm))
        scored.sort(reverse=True)
        return scored[0][1]

    def candidate_from_xy(candidate_id, x, y, state):
        arm_grid = state["arm_grid"]
        scale = arm_grid / 10.24
        ix = max(0, min(arm_grid - 1, int((x + 5.12) * scale)))
        iy = max(0, min(arm_grid - 1, int((y + 5.12) * scale)))
        wall_rng = _random.Random(state["seed"] * 1000003 + int(candidate_id))
        raw = wall_rng.lognormvariate(0.0, 1.05)
        spatial = 1.0 + 0.65 * abs(_math.sin(1.7 * x) * _math.cos(1.3 * y))
        expected_wall_s = max(
            state["min_sleep"],
            min(
                state["max_sleep"],
                state["min_sleep"] + state["sleep_scale"] * raw * spatial,
            ),
        )
        return {
            "candidate_id": int(candidate_id),
            "x": float(x),
            "y": float(y),
            "arm": iy * arm_grid + ix,
            "expected_wall_s": expected_wall_s,
        }

    def candidate_from_arm(candidate_id, arm, state, rng):
        arm_grid = state["arm_grid"]
        ix = int(arm) % arm_grid
        iy = int(arm) // arm_grid
        dx = 10.24 / arm_grid
        x0 = -5.12 + ix * dx
        y0 = -5.12 + iy * dx
        return candidate_from_xy(
            candidate_id,
            rng.uniform(x0, x0 + dx),
            rng.uniform(y0, y0 + dx),
            state,
        )

    def generate_candidate(state, rng):
        cid = state["next_candidate_id"]
        state["next_candidate_id"] += 1

        if rng.random() < 0.10:
            candidate = candidate_from_xy(
                cid,
                rng.uniform(-5.12, 5.12),
                rng.uniform(-5.12, 5.12),
                state,
            )
            source = "global"
        elif state.get("best_result") and rng.random() < 0.22:
            best = state["best_result"]
            candidate = candidate_from_xy(
                cid,
                max(-5.12, min(5.12, rng.gauss(float(best["x"]), 0.55))),
                max(-5.12, min(5.12, rng.gauss(float(best["y"]), 0.55))),
                state,
            )
            source = "local-best"
        else:
            candidate = candidate_from_arm(cid, sample_arm(state, rng), state, rng)
            source = "thompson-arm"

        candidate["source"] = source
        return candidate

    lock_fd = acquire_lock()
    try:
        state = _json.loads(state_path.read_text())
        arm = str(int(result["arm"]))
        reward = float(result["reward"])
        stats = state["arm_stats"][arm]
        stats["n"] += 1
        delta = reward - stats["mean"]
        stats["mean"] += delta / stats["n"]
        stats["m2"] += delta * (reward - stats["mean"])

        if state["best_reward"] is None or reward > state["best_reward"]:
            state["best_reward"] = reward
            state["best_result"] = result

        state["completed_evals"] += 1
        write_event(
            {
                "event": "complete",
                "candidate_id": result["candidate_id"],
                "arm": result["arm"],
                "objective": result["objective"],
                "reward": result["reward"],
                "actual_wall_s": result["actual_wall_s"],
                "best_reward": state["best_reward"],
                "best_candidate_id": state["best_result"]["candidate_id"],
                "completed": state["completed_evals"],
            },
            state,
        )

        elapsed_s = _time.time() - state["start_epoch"]
        open_for_generation = (
            state["autonomous"]
            and state["next_candidate_id"] < state["max_evals"]
            and (state["target_wall_s"] <= 0.0 or elapsed_s < state["target_wall_s"])
        )

        if open_for_generation:
            rng = _random.Random(state["seed"] + 999 + state["completed_evals"] * 7919)
            candidate = generate_candidate(state, rng)
            write_event(
                {
                    "event": "generate",
                    "candidate_id": candidate["candidate_id"],
                    "arm": candidate["arm"],
                    "source": candidate.get("source", ""),
                    "expected_wall_s": candidate["expected_wall_s"],
                    "generated_total": state["next_candidate_id"],
                    "target_wall_s": state["target_wall_s"],
                },
                state,
            )
            state_path.write_text(_json.dumps(state, indent=2))
            return ChoreSpec(
                args=(candidate, state["noise"], state["seed"]),
                kwargs=None,
                resources=Resources(
                    num_tasks=1,
                    cores_per_task=1,
                    gpus_per_task=0,
                    mpi=False,
                    inherit_env=True,
                ),
                qualname="rastrigin-eval",
            )

        state_path.write_text(_json.dumps(state, indent=2))
        return None
    finally:
        _os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def initialize_state(initial_count):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "arm_grid": ARM_GRID,
        "seed": SEED,
        "noise": NOISE,
        "cost_power": COST_POWER,
        "autonomous": AUTONOMOUS_MODE,
        "max_evals": N_CANDIDATES,
        "target_wall_s": TARGET_WALL_S,
        "initial_tasks": INITIAL_TASKS,
        "next_candidate_id": initial_count,
        "completed_evals": 0,
        "min_sleep": MIN_SLEEP,
        "max_sleep": MAX_SLEEP,
        "sleep_scale": SLEEP_SCALE,
        "start_epoch": time.time(),
        "best_reward": None,
        "best_result": None,
        "arm_stats": {
            str(arm): {"n": 0, "mean": -42.0, "m2": 0.0}
            for arm in range(ARM_GRID * ARM_GRID)
        },
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))
    TRACE_PATH.write_text("")


def collect_results(workflow_dir):
    rows = []
    result_paths = Path(workflow_dir).glob("out/chore-rastrigin-eval-*/result.pickle")
    for path in sorted(result_paths):
        with path.open("rb") as fh:
            rows.append(pickle.load(fh))
    rows.sort(key=lambda row: row["candidate_id"])
    return rows


def write_summary(rows):
    completion_events = []
    generation_events = []
    if TRACE_PATH.exists():
        with TRACE_PATH.open() as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("event") == "complete":
                    completion_events.append(event)
                elif event.get("event") == "generate":
                    generation_events.append(event)

    best = min(rows, key=lambda row: row["objective"]) if rows else None
    summary = {
        "workflow_api": "Pipeline.strategy",
        "queue_note": (
            "Spawned chores are admitted with FluxManager._add_chore() and "
            "sorted by nice value; this workflow does not manually reorder the "
            "existing ready queue."
        ),
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
        "n_reorder_events": 0,
    }

    with open("adaptive_ts_results_new_api.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open("adaptive_ts_results_new_api.csv", "w", newline="") as fh:
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

    print("\n=== ADAPTIVE THOMPSON SAMPLING SUMMARY (NEW API) ===")
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
    print("results: adaptive_ts_results_new_api.json adaptive_ts_results_new_api.csv")


initial_count = min(N_CANDIDATES, INITIAL_TASKS) if AUTONOMOUS_MODE else N_CANDIDATES
print(
    "[ATS new API] Building %d initial candidates on %dx%d Rastrigin arms "
    "(autonomous=%s max_evals=%d target_wall_s=%.1f)"
    % (initial_count, ARM_GRID, ARM_GRID, AUTONOMOUS_MODE, N_CANDIDATES, TARGET_WALL_S),
    flush=True,
)

initialize_state(initial_count)
os.environ["MATENSEMBLE_TS_STATE"] = str(STATE_PATH)
os.environ["MATENSEMBLE_TS_TRACE"] = str(TRACE_PATH)

for candidate in build_candidates(initial_count, ARM_GRID, SEED):
    evaluate_candidate(candidate, NOISE, SEED)

t0 = time.perf_counter()
future = pipe.submit(
    buffer_time=0.0,
    log_delay=LOG_DELAY,
    set_cpu_affinity=True,
    set_gpu_affinity=False,
    adaptive=True,
)
future.result()
elapsed = time.perf_counter() - t0

rows = collect_results(pipe._base_dir)
write_summary(rows)
print("workflow elapsed: %.2f s" % elapsed)

status_path = pipe._base_dir / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}
if status.get("failed", 0):
    raise SystemExit(1)
