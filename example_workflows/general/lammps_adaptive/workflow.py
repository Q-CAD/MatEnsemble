from __future__ import annotations

from matensemble.chore import ChoreSpec
from matensemble.model import Resources
from matensemble.pipeline import Pipeline


pipe = Pipeline()

LJ_RESOURCE_SPEC = dict(num_tasks=1, cores_per_task=1)
ANALYSIS_RESOURCE_SPEC = dict(num_tasks=1, cores_per_task=1)

LJ_RESOURCES = Resources(**LJ_RESOURCE_SPEC)


def _run_lj_relaxation(
    *,
    temperature: float,
    seed: int,
    steps: int,
    stage: str,
    write_artifacts: bool = False,
    parent_temperature: float | None = None,
) -> dict:
    """Run a small CPU Lennard-Jones LAMMPS relaxation."""

    from lammps import lammps

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    try:
        for command in [
            "units lj",
            "atom_style atomic",
            "boundary p p p",
            "lattice fcc 0.8442",
            "region box block 0 5 0 5 0 5",
            "create_box 1 box",
            "create_atoms 1 box",
            "mass 1 1.0",
            "pair_style lj/cut 2.5",
            "pair_coeff 1 1 1.0 1.0 2.5",
            "neighbor 0.3 bin",
            "neigh_modify every 1 delay 0 check yes",
            "thermo_modify norm yes",
        ]:
            lmp.command(command)

        lmp.command(
            f"velocity all create {temperature:.6f} {seed} "
            "mom yes rot yes dist gaussian"
        )
        lmp.command("fix integrate all nve")
        lmp.command(f"run {steps}")

        if write_artifacts:
            lmp.command("write_data final.data")
            lmp.command("write_dump all custom final.dump id type x y z")

        return {
            "kind": "lammps_result",
            "stage": stage,
            "temperature": temperature,
            "parent_temperature": parent_temperature,
            "seed": seed,
            "steps": steps,
            "atoms": int(lmp.get_natoms()),
            "pe_per_atom": float(lmp.get_thermo("pe")),
            "ke_per_atom": float(lmp.get_thermo("ke")),
            "final_temperature": float(lmp.get_thermo("temp")),
        }
    finally:
        lmp.close()


@pipe.chore(name="run_lj_screen", **LJ_RESOURCE_SPEC)
def run_lj_screen(temperature: float, seed: int) -> dict:
    """Run an inexpensive LAMMPS screen point."""

    return _run_lj_relaxation(
        temperature=temperature,
        seed=seed,
        steps=40,
        stage="screen",
    )


@pipe.chore(name="score_lj_screen", **ANALYSIS_RESOURCE_SPEC)
def score_lj_screen(lammps_result: dict) -> dict:
    """Score a screen result and decide whether to launch validation."""

    pe = lammps_result["pe_per_atom"]
    validate = lammps_result["stage"] == "screen" and pe <= -5.85
    return {
        "kind": "score",
        "temperature": lammps_result["temperature"],
        "seed": lammps_result["seed"],
        "pe_per_atom": pe,
        "score": -pe,
        "decision": "validate" if validate else "stop",
        "reason": (
            "low relaxed potential energy"
            if validate
            else "screen point did not pass validation threshold"
        ),
    }


@pipe.chore(name="validate_lj_state", **LJ_RESOURCE_SPEC)
def validate_lj_state(score: dict) -> dict:
    """Rerun a promising state longer with a different seed."""

    return _run_lj_relaxation(
        temperature=max(0.1, score["temperature"] + 0.05),
        seed=score["seed"] + 1009,
        steps=80,
        stage="validation",
        write_artifacts=True,
        parent_temperature=score["temperature"],
    )


@pipe.strategy(
    bolo_list=["score_lj_screen"],
    name="request_validation",
    **ANALYSIS_RESOURCE_SPEC,
)
def request_validation(score: dict) -> ChoreSpec | None:
    """Spawn a validation LAMMPS chore for promising score results."""

    if score["decision"] != "validate":
        return None

    return ChoreSpec(
        args=(score,),
        kwargs=None,
        qualname="validate_lj_state",
        resources=LJ_RESOURCES,
    )


for index, temperature in enumerate([0.70, 1.00, 1.30, 1.60], start=1):
    screen = run_lj_screen(temperature, seed=87000 + index)
    score_lj_screen(screen)


future = pipe.submit(log_delay=1)
results = future.result()

scores = [
    result
    for result in results.values()
    if isinstance(result, dict) and result.get("kind") == "score"
]
validations = [
    result
    for result in results.values()
    if isinstance(result, dict)
    and result.get("kind") == "lammps_result"
    and result.get("stage") == "validation"
]

print(f"screened_temperatures={sorted(score['temperature'] for score in scores)}")
print(
    "validated_from_temperatures="
    f"{sorted(result['parent_temperature'] for result in validations)}"
)
print(
    "best_screen_pe_per_atom="
    f"{min(score['pe_per_atom'] for score in scores):.6f}"
)

if not validations:
    raise RuntimeError("Expected at least one adaptive LAMMPS validation chore.")
