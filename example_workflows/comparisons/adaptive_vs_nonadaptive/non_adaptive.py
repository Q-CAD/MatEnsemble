from __future__ import annotations

import pprint
import time

from matensemble.pipeline import Pipeline

pipe = Pipeline()

SLEEP_SECONDS = {
    "short": 1,
    "short_mid": 25,
    "mid": 50,
    "mid_long": 75,
    "long": 100,
}

# 100 chores total, weighted toward the short cases.
WORKLOAD_COUNTS = {
    "short": 55,
    "short_mid": 20,
    "mid": 12,
    "mid_long": 8,
    "long": 5,
}

WORKLOAD = (kind for kind, count in WORKLOAD_COUNTS.items() for _ in range(count))


def _run(kind: str) -> dict[str, float | str]:
    started = time.perf_counter()
    time.sleep(SLEEP_SECONDS[kind])
    return {
        "kind": kind,
        "target_seconds": float(SLEEP_SECONDS[kind]),
        "elapsed_seconds": time.perf_counter() - started,
    }


@pipe.chore()
def short() -> dict[str, float | str]:
    return _run("short")


@pipe.chore()
def short_mid() -> dict[str, float | str]:
    return _run("short_mid")


@pipe.chore()
def mid() -> dict[str, float | str]:
    return _run("mid")


@pipe.chore()
def mid_long() -> dict[str, float | str]:
    return _run("mid_long")


@pipe.chore()
def long() -> dict[str, float | str]:
    return _run("long")


CHORES = {
    "short": short,
    "short_mid": short_mid,
    "mid": mid,
    "mid_long": mid_long,
    "long": long,
}


for kind in WORKLOAD:
    CHORES[kind]()

future = pipe.submit(log_delay=1, adaptive=False)
results = future.result()

completed = [
    result
    for result in results.values()
    if isinstance(result, dict) and result.get("kind") in WORKLOAD_COUNTS
]

print(f"adaptive benchmark completed {len(completed)} chores")
pprint.pprint(results)
