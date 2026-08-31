"""Visualize and model the benefit of adaptive workflow scheduling.

The measured comparison supplies one benchmark point.  The prediction curve
holds its task count, worker capacity, and task order fixed while progressively
stretching the empirical task-duration distribution.  This isolates the effect
of runtime spread on adaptive backfilling versus nonadaptive wave scheduling.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ADAPTIVE_RUN = HERE / "adaptive_run"
DEFAULT_NONADAPTIVE_RUN = HERE / "non_adaptive_run"
DEFAULT_WORKLOAD_SCRIPT = HERE / "adaptive.py"
DEFAULT_OUTPUT_DIR = HERE / "results"

ADAPTIVE_COLOR = "#007F80"
NONADAPTIVE_COLOR = "#40566E"
ACCENT_COLOR = "#E26D5A"
GRID_COLOR = "#DCE3E8"
TEXT_COLOR = "#24313A"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-run",
        type=Path,
        default=DEFAULT_ADAPTIVE_RUN,
        help="completed adaptive workflow directory",
    )
    parser.add_argument(
        "--nonadaptive-run",
        type=Path,
        default=DEFAULT_NONADAPTIVE_RUN,
        help="completed nonadaptive workflow directory",
    )
    parser.add_argument(
        "--workload-script",
        type=Path,
        default=DEFAULT_WORKLOAD_SCRIPT,
        help="script containing literal SLEEP_SECONDS and WORKLOAD_COUNTS mappings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the PNG, SVG, CSV, and JSON outputs",
    )
    parser.add_argument(
        "--max-stretch",
        type=float,
        default=2.5,
        help="largest empirical duration-stretch exponent (default: 2.5)",
    )
    return parser.parse_args(argv)


def workflow_elapsed_seconds(run_dir: Path) -> float:
    status_path = run_dir / "status.json"
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)

    workflow = status.get("workflow", {})
    if workflow.get("state") != "completed":
        raise ValueError(f"workflow is not completed: {status_path}")
    return float(workflow["elapsed_seconds"])


def workflow_capacity(run_dir: Path) -> int:
    """Infer the one-core worker capacity from the status history."""
    history_path = run_dir / "status_history.jsonl"
    capacities: list[int] = []
    with history_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            capacities.append(int(record["free_cores"]) + int(record["running"]))

    if not capacities:
        raise ValueError(f"status history is empty: {history_path}")
    return max(capacities)


def literal_assignment(script_path: Path, name: str) -> dict[str, int]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return {str(key): int(item) for key, item in value.items()}
    raise ValueError(f"could not find a literal {name} mapping in {script_path}")


def target_durations(script_path: Path) -> np.ndarray:
    sleep_seconds = literal_assignment(script_path, "SLEEP_SECONDS")
    workload_counts = literal_assignment(script_path, "WORKLOAD_COUNTS")
    missing = workload_counts.keys() - sleep_seconds.keys()
    if missing:
        raise ValueError(f"workload kinds are missing durations: {sorted(missing)}")

    durations = [
        float(sleep_seconds[kind])
        for kind, count in workload_counts.items()
        for _ in range(count)
    ]
    if not durations or min(durations) <= 0:
        raise ValueError("target durations must be nonempty and positive")
    return np.asarray(durations, dtype=float)


def adaptive_makespan(durations: np.ndarray, workers: int) -> float:
    """Model immediate backfill using FIFO list scheduling."""
    loads = np.zeros(workers, dtype=float)
    for duration in durations:
        loads[int(np.argmin(loads))] += duration
    return float(np.max(loads))


def nonadaptive_makespan(durations: np.ndarray, workers: int) -> float:
    """Model fixed waves, each gated by its slowest task."""
    return float(
        sum(
            np.max(durations[start : start + workers])
            for start in range(0, len(durations), workers)
        )
    )


def duration_spread(durations: np.ndarray) -> float:
    """Return the population coefficient of variation, sigma / mean."""
    return float(np.std(durations) / np.mean(durations))


def prediction_curve(
    durations: np.ndarray,
    workers: int,
    observed_reduction: float,
    max_stretch: float,
) -> tuple[list[dict[str, float | str]], float]:
    if max_stretch < 1.0:
        raise ValueError("max-stretch must be at least 1.0 to include the benchmark")

    # Include exponent 1.0 exactly so the empirical workload is an explicit
    # point in the modeled family.
    exponents = np.unique(
        np.concatenate((np.linspace(0.0, max_stretch, 301), np.asarray([1.0])))
    )
    exponents.sort()

    mean_duration = float(np.mean(durations))
    raw_rows: list[dict[str, float | str]] = []
    baseline_model_reduction: float | None = None

    for exponent in exponents:
        stretched = durations**exponent
        stretched *= mean_duration / float(np.mean(stretched))
        adaptive_seconds = adaptive_makespan(stretched, workers)
        nonadaptive_seconds = nonadaptive_makespan(stretched, workers)
        reduction = 100.0 * (nonadaptive_seconds - adaptive_seconds) / nonadaptive_seconds

        if np.isclose(exponent, 1.0):
            baseline_model_reduction = reduction

        raw_rows.append(
            {
                "stretch_exponent": float(exponent),
                "coefficient_of_variation": duration_spread(stretched),
                "adaptive_model_seconds": adaptive_seconds,
                "nonadaptive_model_seconds": nonadaptive_seconds,
                "uncalibrated_reduction_percent": reduction,
                "regime": "interpolation" if exponent <= 1.0 else "extrapolation",
            }
        )

    if baseline_model_reduction is None or baseline_model_reduction <= 0:
        raise ValueError("the empirical workload does not produce a schedulable benefit")

    # One observed comparison is available.  Scale the idealized scheduling
    # model by that point to account for the benchmark's launch/poll overhead.
    calibration_factor = observed_reduction / baseline_model_reduction
    for row in raw_rows:
        row["predicted_reduction_percent"] = (
            float(row["uncalibrated_reduction_percent"]) * calibration_factor
        )

    return raw_rows, calibration_factor


def write_curve_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    fieldnames = [
        "stretch_exponent",
        "coefficient_of_variation",
        "adaptive_model_seconds",
        "nonadaptive_model_seconds",
        "uncalibrated_reduction_percent",
        "predicted_reduction_percent",
        "regime",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    adaptive_seconds: float,
    nonadaptive_seconds: float,
    observed_spread: float,
    rows: list[dict[str, float | str]],
    workers: int,
    task_count: int,
    output_dir: Path,
) -> None:
    observed_reduction = 100.0 * (
        nonadaptive_seconds - adaptive_seconds
    ) / nonadaptive_seconds
    throughput_multiplier = nonadaptive_seconds / adaptive_seconds
    saved_seconds = nonadaptive_seconds - adaptive_seconds

    x = np.asarray([row["coefficient_of_variation"] for row in rows], dtype=float)
    y = np.asarray([row["predicted_reduction_percent"] for row in rows], dtype=float)
    exponent = np.asarray([row["stretch_exponent"] for row in rows], dtype=float)
    interpolation = exponent <= 1.0
    extrapolation = exponent >= 1.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )
    fig, (ax_bars, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.8),
        gridspec_kw={"width_ratios": (0.88, 1.45), "wspace": 0.28},
    )
    fig.patch.set_facecolor("white")

    # Panel A: measured benchmark.
    labels = ["Nonadaptive\n(waves)", "Adaptive\n(backfill)"]
    times = [nonadaptive_seconds, adaptive_seconds]
    colors = [NONADAPTIVE_COLOR, ADAPTIVE_COLOR]
    y_positions = [1, 0]
    bars = ax_bars.barh(y_positions, times, height=0.52, color=colors, zorder=3)
    for bar, seconds in zip(bars, times):
        ax_bars.text(
            seconds - 4.0,
            bar.get_y() + bar.get_height() / 2,
            f"{seconds:.1f} s",
            va="center",
            ha="right",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    ax_bars.set_yticks(y_positions, labels)
    ax_bars.set_xlabel("Workflow wall-clock time (seconds)")
    # Leave a right-hand margin for the headline metrics so they do not obscure
    # the measured bars or their labels.
    ax_bars.set_xlim(0, nonadaptive_seconds * 1.42)
    ax_bars.set_ylim(-0.8, 1.65)
    ax_bars.text(
        0.0,
        1.04,
        f"{task_count} tasks on {workers} worker slots",
        transform=ax_bars.transAxes,
        color="#62727D",
        fontsize=9.5,
        va="bottom",
    )
    ax_bars.annotate(
        "",
        xy=(adaptive_seconds, -0.55),
        xytext=(nonadaptive_seconds, -0.55),
        arrowprops={"arrowstyle": "<->", "color": ACCENT_COLOR, "lw": 1.8},
    )
    ax_bars.text(
        (adaptive_seconds + nonadaptive_seconds) / 2,
        -0.72,
        f"{saved_seconds:.1f} s saved",
        color=ACCENT_COLOR,
        ha="center",
        va="top",
        fontweight="bold",
    )
    ax_bars.text(
        0.96,
        0.88,
        f"{observed_reduction:.1f}% lower\n{throughput_multiplier:.2f}× throughput",
        transform=ax_bars.transAxes,
        ha="right",
        va="top",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFF4F0",
            "edgecolor": "none",
        },
    )

    # Panel B: one-point-calibrated scheduling model.
    ax_curve.fill_between(x, 0, y, color=ADAPTIVE_COLOR, alpha=0.08, zorder=1)
    ax_curve.plot(
        x[interpolation],
        y[interpolation],
        color=ADAPTIVE_COLOR,
        linewidth=3.0,
        label="Calibrated prediction",
        zorder=3,
    )
    ax_curve.plot(
        x[extrapolation],
        y[extrapolation],
        color=ADAPTIVE_COLOR,
        linewidth=3.0,
        linestyle=(0, (4, 3)),
        zorder=3,
    )
    ax_curve.axvline(
        observed_spread,
        color=ACCENT_COLOR,
        linewidth=1.2,
        linestyle=(0, (2, 3)),
        alpha=0.8,
        zorder=2,
    )
    ax_curve.scatter(
        [observed_spread],
        [observed_reduction],
        s=115,
        color=ACCENT_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=5,
        label="Observed run",
    )
    ax_curve.annotate(
        f"Current workload\nCV = {observed_spread:.2f}  •  {observed_reduction:.1f}% reduction",
        xy=(observed_spread, observed_reduction),
        xytext=(14, -45),
        textcoords="offset points",
        color=TEXT_COLOR,
        fontsize=9.5,
        arrowprops={"arrowstyle": "-", "color": ACCENT_COLOR, "lw": 1.2},
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
        },
    )
    ax_curve.text(
        0.98,
        0.08,
        "Dashed segment = extrapolation\nbeyond the measured spread",
        transform=ax_curve.transAxes,
        ha="right",
        va="bottom",
        color="#62727D",
        fontsize=9,
    )
    ax_curve.set_xlabel("Task-runtime spread (coefficient of variation, σ / μ)")
    ax_curve.set_ylabel("Predicted wall-clock reduction (%)")
    ax_curve.set_xlim(0, float(np.max(x)) * 1.03)
    ax_curve.set_ylim(0, float(np.max(y)) * 1.18)
    ax_curve.text(
        0.0,
        1.04,
        "Empirical duration distribution progressively stretched",
        transform=ax_curve.transAxes,
        color="#62727D",
        fontsize=9.5,
        va="bottom",
    )
    ax_curve.legend(loc="upper left", frameon=False, fontsize=9.5)

    for axis in (ax_bars, ax_curve):
        axis.grid(axis="x" if axis is ax_bars else "both", color=GRID_COLOR, lw=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID_COLOR)
        axis.spines["bottom"].set_color(GRID_COLOR)

    ax_bars.text(
        -0.15,
        1.13,
        "A",
        transform=ax_bars.transAxes,
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax_curve.text(
        -0.10,
        1.13,
        "B",
        transform=ax_curve.transAxes,
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax_bars.text(
        0.0,
        1.13,
        "Observed benchmark",
        transform=ax_bars.transAxes,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        va="bottom",
    )
    ax_curve.text(
        0.0,
        1.13,
        "Adaptive benefit grows with runtime spread",
        transform=ax_curve.transAxes,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        va="bottom",
    )

    fig.suptitle(
        "Adaptive scheduling turns straggler time into useful throughput",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.07,
        0.925,
        "Immediate slot refill avoids waiting for every task in a nonadaptive wave to finish.",
        ha="left",
        fontsize=11,
        color="#62727D",
    )
    fig.text(
        0.07,
        0.018,
        "Reduction = (Tnonadaptive − Tadaptive) / Tnonadaptive × 100.  "
        "Prediction holds task count, order, mean duration, and worker capacity fixed.",
        ha="left",
        fontsize=8.8,
        color="#62727D",
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.09, right=0.98)

    fig.savefig(output_dir / "adaptive_benefit.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "adaptive_benefit.svg", facecolor="white")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    adaptive_seconds = workflow_elapsed_seconds(args.adaptive_run)
    nonadaptive_seconds = workflow_elapsed_seconds(args.nonadaptive_run)

    adaptive_capacity = workflow_capacity(args.adaptive_run)
    nonadaptive_capacity = workflow_capacity(args.nonadaptive_run)
    if adaptive_capacity != nonadaptive_capacity:
        raise ValueError(
            "runs used different capacities: "
            f"adaptive={adaptive_capacity}, nonadaptive={nonadaptive_capacity}"
        )

    durations = target_durations(args.workload_script)
    observed_spread = duration_spread(durations)
    observed_reduction = 100.0 * (
        nonadaptive_seconds - adaptive_seconds
    ) / nonadaptive_seconds
    rows, calibration_factor = prediction_curve(
        durations,
        adaptive_capacity,
        observed_reduction,
        args.max_stretch,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_curve_csv(rows, args.output_dir / "adaptive_benefit_curve.csv")
    make_figure(
        adaptive_seconds,
        nonadaptive_seconds,
        observed_spread,
        rows,
        adaptive_capacity,
        len(durations),
        args.output_dir,
    )

    summary = {
        "adaptive_elapsed_seconds": adaptive_seconds,
        "nonadaptive_elapsed_seconds": nonadaptive_seconds,
        "wall_clock_seconds_saved": nonadaptive_seconds - adaptive_seconds,
        "wall_clock_reduction_percent": observed_reduction,
        "throughput_multiplier": nonadaptive_seconds / adaptive_seconds,
        "task_runtime_coefficient_of_variation": observed_spread,
        "task_count": len(durations),
        "worker_slots": adaptive_capacity,
        "prediction_calibration_factor": calibration_factor,
    }
    with (args.output_dir / "adaptive_benefit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"adaptive={adaptive_seconds:.3f}s nonadaptive={nonadaptive_seconds:.3f}s "
        f"reduction={observed_reduction:.2f}% CV={observed_spread:.3f}"
    )
    print(f"wrote visualization and data to {args.output_dir}")


if __name__ == "__main__":
    main()
