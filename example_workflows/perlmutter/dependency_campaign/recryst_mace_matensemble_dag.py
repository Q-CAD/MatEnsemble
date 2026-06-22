import argparse
import json
import os
import sys
import types
from pathlib import Path

def _ensure_h5py_stub():
    # This container's h5py can crash during MACE imports. The MACE inference
    # path used by LAMMPS does not need h5py, so follow Kaleb's working template
    # and force a stub before every model-load path.
    sys.modules["h5py"] = types.ModuleType("h5py")


_ensure_h5py_stub()

import lammps
import lammps.mliap
from mace.calculators.lammps_mliap_mace import MACEEdgeForcesWrapper as _MACEWrapper
from matensemble.pipeline import Pipeline

if not hasattr(_MACEWrapper, "total_charge"):
    _MACEWrapper.total_charge = 0.0

KB_EV_PER_K = 8.617333262145e-5
EV_J = 1.602176634e-19
HBAR_EV_S = 6.582119569e-16
AMU_KG = 1.66053906660e-27
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / os.environ.get("RECRYST_CONFIG", "recryst_config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)

TASKS_PER_CHORE = int(os.environ.get("RECRYST_TASKS_PER_CHORE", CONFIG.get("tasks_per_chore", 4)))
GPUS_PER_TASK = int(os.environ.get("RECRYST_GPUS_PER_TASK", CONFIG.get("gpus_per_task", 1)))
CORES_PER_TASK = int(os.environ.get("RECRYST_CORES_PER_TASK", CONFIG.get("cores_per_task", 1)))
LAMMPS_KOKKOS_THREADS = max(1, int(os.environ.get(
    "LAMMPS_KOKKOS_THREADS",
    os.environ.get("OMP_NUM_THREADS", "1"),
)))
LAMMPS_GPUS_PER_NODE = int(os.environ.get(
    "RECRYST_LAMMPS_GPUS_PER_NODE",
    CONFIG.get("lammps_gpus_per_node", 4),
))

pipe = Pipeline()


def _rank0():
    for key in ("FLUX_TASK_RANK", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return value == "0"
    return True


def _abspath(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _case_by_name(name):
    for case in CONFIG["cases"]:
        if case["name"] == name:
            return case
    raise ValueError(f"Unknown case: {name}")


def _defaults(case):
    merged = dict(CONFIG["defaults"])
    merged.update(case.get("settings", {}))
    return merged


def _type_list(values):
    return " ".join(str(v) for v in values)


def _init_lammps(work_dir, use_kokkos=True):
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    if use_kokkos:
        cmdargs = [
            "-k", "on", "t", str(LAMMPS_KOKKOS_THREADS), "g", str(LAMMPS_GPUS_PER_NODE),
            "-sf", "kk",
            "-pk", "kokkos", "neigh", "half", "newton", "off",
            "-echo", "both",
            "-log", "log.lammps",
        ]
    else:
        cmdargs = [
            "-echo", "both",
            "-log", "log.lammps",
        ]
    lmp = lammps.lammps(cmdargs=cmdargs)
    if use_kokkos:
        lammps.mliap.activate_mliappy_kokkos(lmp)
    else:
        lammps.mliap.activate_mliappy(lmp)
    return lmp


def _base_system(lmp, case, model_path, pair_scale_variable=None, use_kokkos=True):
    settings = _defaults(case)
    replicate = settings["replicate"]
    _ensure_h5py_stub()

    lmp.commands_string("""
units         metal
atom_style    atomic
atom_modify   map yes
newton        on
boundary      p p f
processors    * * 1
neighbor      2.0 bin
neigh_modify  every 1 delay 0 check yes
""")
    lmp.command(f"read_data {_abspath(case['structure'])}")
    if replicate != [1, 1, 1]:
        lmp.command(f"replicate {replicate[0]} {replicate[1]} {replicate[2]}")
    if pair_scale_variable:
        hybrid_style = "hybrid/scaled/kk" if use_kokkos else "hybrid/scaled"
        lmp.command(f"pair_style {hybrid_style} v_{pair_scale_variable} mliap unified {model_path} 0")
        lmp.command(f"pair_coeff * * mliap {' '.join(case['species'])}")
    else:
        lmp.command(f"pair_style mliap unified {model_path} 0")
        lmp.command(f"pair_coeff * * {' '.join(case['species'])}")
    lmp.command(f"timestep {settings['timestep_ps']}")
    lmp.command(f"thermo {settings['thermo_every']}")
    lmp.command("thermo_style custom step temp pe ke etotal press")


def _define_film_groups(lmp, case):
    lmp.command(f"group film_type type {_type_list(case['film_types'])}")
    if "film_z_min" in case and "film_z_max" in case:
        lmp.command(f"region film_z block INF INF INF INF {case['film_z_min']} {case['film_z_max']} units box")
    elif "film_z_min" in case:
        lmp.command(f"region film_z block INF INF INF INF {case['film_z_min']} INF units box")
    elif "film_z_max" in case:
        lmp.command(f"region film_z block INF INF INF INF INF {case['film_z_max']} units box")
    else:
        lmp.command("region film_z block INF INF INF INF INF INF units box")
    lmp.command("group film_z region film_z")
    lmp.command("group film intersect film_type film_z")
    lmp.command("group substrate subtract all film")


def _film_surface_wall_side(case, settings):
    side = case.get("melt_surface_wall_side", settings.get("melt_surface_wall_side", "auto"))
    if side in (None, "none", "off", "false"):
        return None
    if side == "auto":
        if "film_z_min" in case and "film_z_max" not in case:
            return "zhi"
        if "film_z_max" in case and "film_z_min" not in case:
            return "zlo"
        return "zhi"
    if side not in ("zlo", "zhi"):
        raise ValueError(f"melt_surface_wall_side must be auto, zlo, zhi, or none; got {side!r}")
    return side


def _add_melt_surface_wall(lmp, case, settings):
    if not settings.get("melt_surface_wall_enabled", True):
        return {"enabled": False}

    side = _film_surface_wall_side(case, settings)
    if side is None:
        return {"enabled": False}

    buffer_a = float(settings.get("melt_surface_wall_buffer_A", 3.0))
    boxlo, boxhi, *_ = lmp.extract_box()
    if side == "zhi":
        lmp.command("variable recryst_film_surface equal bound(film,zmax)")
        surface_z = float(lmp.extract_variable("recryst_film_surface", None, 0))
        wall_z = min(surface_z + buffer_a, float(boxhi[2]))
    else:
        lmp.command("variable recryst_film_surface equal bound(film,zmin)")
        surface_z = float(lmp.extract_variable("recryst_film_surface", None, 0))
        wall_z = max(surface_z - buffer_a, float(boxlo[2]))

    fix_id = "melt_surface_cap"
    lmp.command(f"fix {fix_id} film wall/reflect {side} {wall_z:.12g}")
    lmp.command("variable recryst_film_surface delete")
    return {
        "enabled": True,
        "fix_id": fix_id,
        "side": side,
        "surface_z_A": surface_z,
        "wall_z_A": wall_z,
        "buffer_A": buffer_a,
        "applied_to_group": "film",
        "stage": "melt_only",
    }


def _add_dump(lmp, filename, every):
    lmp.command(f"dump traj all custom {every} {filename} id type x y z vx vy vz fx fy fz")
    lmp.command("dump_modify traj sort id")


def _write_metadata(work_dir, payload):
    if _rank0():
        with (Path(work_dir) / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _read_lammps_data_stats(path):
    masses = {}
    counts = {}
    xlo = xhi = ylo = yhi = zlo = zhi = None
    section = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
                xlo, xhi = float(parts[0]), float(parts[1])
                continue
            if len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
                ylo, yhi = float(parts[0]), float(parts[1])
                continue
            if len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
                zlo, zhi = float(parts[0]), float(parts[1])
                continue
            if line.startswith("Masses"):
                section = "masses"
                continue
            if line.startswith("Atoms"):
                section = "atoms"
                continue
            if section == "masses":
                try:
                    type_id = int(parts[0])
                    masses[type_id] = float(parts[1])
                except (ValueError, IndexError):
                    pass
                continue
            if section == "atoms":
                try:
                    int(parts[0])
                    type_id = int(parts[1])
                except (ValueError, IndexError):
                    continue
                counts[type_id] = counts.get(type_id, 0) + 1

    volume = None
    if None not in (xlo, xhi, ylo, yhi, zlo, zhi):
        volume = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)

    return {
        "masses": masses,
        "counts": counts,
        "natoms": sum(counts.values()),
        "volume": volume,
    }


def _load_spring_constants(path, species):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    constants = data.get("spring_constants_by_species", {})
    missing = [name for name in species if constants.get(name) is None]
    if missing:
        raise ValueError(f"Missing calibrated spring constants for species: {missing}")
    return {name: float(constants[name]) for name in species}


def _trapz_xy(rows):
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(rows[:-1], rows[1:]):
        y0, x0 = left
        y1, x1 = right
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


def _load_switching_file(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    return rows


def _state_source(case_name, state, seed):
    if state == "crystal":
        return ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "crystal_300K.lmp"
    if state == "amorphous":
        return ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "amorphous_300K.lmp"
    raise ValueError(f"Unknown state: {state}")


def _postprocess_frenkel_ladd(work_dir, source, case, springs, temperature, settings=None):
    stats = _read_lammps_data_stats(source)
    species = case["species"]
    harmonic = 0.0
    k_total = 0.0
    harmonic_by_species = {}

    for type_id, name in enumerate(species, start=1):
        count = stats["counts"].get(type_id, 0)
        mass = stats["masses"].get(type_id)
        spring = springs[name]
        if count == 0:
            harmonic_by_species[name] = 0.0
            continue
        if mass is None:
            raise ValueError(f"No mass found for type {type_id} ({name}) in {source}")
        omega = ((spring * EV_J) / (mass * AMU_KG)) ** 0.5 * 1.0e10
        contribution = 3.0 * count * KB_EV_PER_K * temperature * (
            __import__("math").log(HBAR_EV_S * omega / (KB_EV_PER_K * temperature))
        )
        harmonic_by_species[name] = contribution
        harmonic += contribution
        k_total += count * spring

    forward = _load_switching_file(Path(work_dir) / "forward.dat")
    backward = _load_switching_file(Path(work_dir) / "backward.dat")
    i_forward = _trapz_xy(forward)
    i_backward = _trapz_xy(backward)
    reversible_work = 0.5 * (i_forward - i_backward)

    f_cm = None
    if stats["natoms"] and stats["volume"] and k_total > 0.0:
        import math
        f_cm = KB_EV_PER_K * temperature * math.log(
            (stats["natoms"] / stats["volume"]) *
            ((2.0 * math.pi * KB_EV_PER_K * temperature / k_total) ** 1.5)
        )

    f_no_cm = harmonic + reversible_work
    f_with_cm = f_no_cm + f_cm if f_cm is not None else None
    natoms = stats["natoms"]
    result = {
        "case": case["name"],
        "temperature_K": temperature,
        "natoms": natoms,
        "volume_angstrom3": stats["volume"],
        "run_settings": settings or {},
        "spring_constants_eV_per_angstrom2": springs,
        "lambda_integrals_eV": {
            "forward": i_forward,
            "backward": i_backward,
            "reversible_work": reversible_work
        },
        "harmonic_reference_eV": {
            "total": harmonic,
            "by_species": harmonic_by_species
        },
        "center_of_mass_correction_eV": f_cm,
        "free_energy_eV": {
            "without_cm_correction": f_no_cm,
            "with_cm_correction_estimate": f_with_cm
        },
        "free_energy_eV_per_atom": {
            "without_cm_correction": f_no_cm / natoms if natoms else None,
            "with_cm_correction_estimate": f_with_cm / natoms if f_with_cm is not None and natoms else None
        },
        "notes": [
            "Frenkel-Ladd integration follows dE = pe - E_harm from the Freitas LAMMPS example.",
            "The COM correction is the single-spring Freitas form generalized with K_total=sum_i N_i K_i; for p p f slabs with vacuum this should be treated as an estimate."
        ]
    }
    with (Path(work_dir) / "frenkel_ladd_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _load_rs_path(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append({
                    "scaled_potential_energy_eV_per_atom": float(parts[0]),
                    "lambda_Tref_over_T": float(parts[1]),
                })
    return rows


def _temperature_grid(settings, t_ref, t_high):
    if "rs_temperature_grid_K" in settings:
        return [float(t) for t in settings["rs_temperature_grid_K"]]

    step = float(settings.get("rs_temperature_grid_step_K", 100.0))
    grid = []
    t = t_ref
    while t <= t_high + 1.0e-8:
        grid.append(round(t, 10))
        t += step
    if abs(grid[-1] - t_high) > 1.0e-8:
        grid.append(t_high)
    return grid


def _cumtrapz(y_values, x_values):
    if not y_values:
        return []
    values = [0.0]
    total = 0.0
    for i in range(1, len(y_values)):
        total += 0.5 * (y_values[i - 1] + y_values[i]) * (x_values[i] - x_values[i - 1])
        values.append(total)
    return values


def _interp_xy(x_values, y_values, x_target):
    pairs = sorted(zip(x_values, y_values), key=lambda item: item[0])
    if x_target <= pairs[0][0]:
        return pairs[0][1]
    if x_target >= pairs[-1][0]:
        return pairs[-1][1]
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        if x0 <= x_target <= x1:
            if abs(x1 - x0) < 1.0e-12:
                return y1
            weight = (x_target - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return pairs[-1][1]


def _freitas_rs_curve(rows, t_ref, f0_per_atom):
    import math
    lambdas = [row["lambda_Tref_over_T"] for row in rows]
    unscaled_pe = [
        row["scaled_potential_energy_eV_per_atom"] / row["lambda_Tref_over_T"]
        for row in rows
    ]
    integrals = _cumtrapz(unscaled_pe, lambdas)
    curve = []
    for lambda_value, work_integral in zip(lambdas, integrals):
        temperature = t_ref / lambda_value
        free_energy = (
            f0_per_atom / lambda_value +
            1.5 * KB_EV_PER_K * temperature * math.log(lambda_value) +
            work_integral / lambda_value
        )
        curve.append({
            "lambda_Tref_over_T": lambda_value,
            "temperature_K": temperature,
            "work_integral_eV_per_atom": work_integral,
            "free_energy_eV_per_atom": free_energy,
        })
    return curve


def _postprocess_reversible_scaling(work_dir, source, case, state, seed, settings, fl_result_path=None):
    stats = _read_lammps_data_stats(source)
    fl_path = Path(fl_result_path) if fl_result_path else (
        ROOT / CONFIG["output_dir"] / case["name"] / state / f"seed_{seed}" / "frenkel_ladd" / "frenkel_ladd_result.json"
    )
    with fl_path.open("r", encoding="utf-8") as handle:
        fl_result = json.load(handle)

    free_energy = fl_result["free_energy_eV"].get("with_cm_correction_estimate")
    free_energy_key = "with_cm_correction_estimate"
    if free_energy is None:
        free_energy = fl_result["free_energy_eV"]["without_cm_correction"]
        free_energy_key = "without_cm_correction"

    t_ref = float(fl_result["temperature_K"])
    t_high = float(settings.get("rs_high_temperature_K", settings.get("melt_temperature_K", t_ref)))
    natoms = stats["natoms"]
    f0_per_atom = free_energy / natoms
    forward_rows = _load_rs_path(Path(work_dir) / "rs_forward.dat")
    backward_rows = _load_rs_path(Path(work_dir) / "rs_backward.dat")
    forward_curve = _freitas_rs_curve(forward_rows, t_ref, f0_per_atom)
    backward_curve = _freitas_rs_curve(list(reversed(backward_rows)), t_ref, f0_per_atom)
    grid = _temperature_grid(settings, t_ref, t_high)

    curve = []
    for temperature in grid:
        lambda_value = t_ref / temperature
        f_forward = _interp_xy(
            [point["lambda_Tref_over_T"] for point in forward_curve],
            [point["free_energy_eV_per_atom"] for point in forward_curve],
            lambda_value,
        )
        f_backward = _interp_xy(
            [point["lambda_Tref_over_T"] for point in backward_curve],
            [point["free_energy_eV_per_atom"] for point in backward_curve],
            lambda_value,
        )
        f_reversible = 0.5 * (f_forward + f_backward)
        hysteresis = f_forward - f_backward
        curve.append({
            "temperature_K": temperature,
            "lambda_Tref_over_T": lambda_value,
            "forward_free_energy_eV_per_atom": f_forward,
            "backward_free_energy_eV_per_atom": f_backward,
            "reversible_free_energy_eV_per_atom": f_reversible,
            "hysteresis_eV_per_atom": hysteresis,
            "reversible_free_energy_eV": f_reversible * natoms if natoms else None,
            "hysteresis_eV": hysteresis * natoms if natoms else None,
        })

    result = {
        "case": case["name"],
        "state": state,
        "seed": seed,
        "natoms": natoms,
        "volume_angstrom3": stats["volume"],
        "reference": {
            "temperature_K": t_ref,
            "frenkel_ladd_result": str(fl_path),
            "free_energy_key": free_energy_key,
            "free_energy_eV": free_energy,
            "free_energy_eV_per_atom": f0_per_atom,
        },
        "run_settings": settings,
        "method": {
            "name": "freitas_reversible_scaling",
            "reference": "Freitas, Asta, and de Koning nonequilibrium reversible-scaling path",
            "lambda": "T_ref/T",
            "integrand": "unscaled potential energy per atom, U(lambda)/lambda",
            "formula": "F(T_ref/lambda)=F0/lambda + 3/2 kB T ln(lambda) + W(lambda)",
        },
        "paths": {
            "forward_samples": len(forward_rows),
            "backward_samples": len(backward_rows),
            "forward_lambda_range": [
                forward_rows[0]["lambda_Tref_over_T"],
                forward_rows[-1]["lambda_Tref_over_T"],
            ] if forward_rows else None,
            "backward_lambda_range": [
                backward_rows[0]["lambda_Tref_over_T"],
                backward_rows[-1]["lambda_Tref_over_T"],
            ] if backward_rows else None,
        },
        "free_energy_curve": curve,
        "notes": [
            "The RS simulation scales the potential-energy surface by lambda=T_ref/T at fixed thermostat temperature T_ref.",
            "The FL result is used as the absolute F0 reference in post-processing.",
            "Large forward/backward hysteresis or structural transformation during the temperature path means the curve should be treated as a nonequilibrium scout."
        ]
    }
    with (Path(work_dir) / "rs_free_energy_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


@pipe.chore(num_tasks=TASKS_PER_CHORE, cores_per_task=CORES_PER_TASK, gpus_per_task=GPUS_PER_TASK)
def prepare_amorphous(case_name: str, seed: int):
    case = _case_by_name(case_name)
    settings = _defaults(case)
    model_path = _abspath(CONFIG["model"])
    work_dir = ROOT / CONFIG["output_dir"] / case_name / "amorphous" / f"seed_{seed}"

    lmp = _init_lammps(work_dir)
    _base_system(lmp, case, model_path)
    _define_film_groups(lmp, case)
    surface_wall = _add_melt_surface_wall(lmp, case, settings)
    _add_dump(lmp, "amorphous.lammpstrj", settings["dump_every"])

    melt_T = settings["melt_temperature_K"]
    equil_T = settings["equil_temperature_K"]
    damp = settings["langevin_damp_ps"]
    lmp.command(f"velocity film create {melt_T} {seed} mom yes rot yes dist gaussian")
    lmp.command("fix int all nve")
    lmp.command(f"fix hot film langevin {melt_T} {melt_T} {damp} {seed + 17} zero yes")
    lmp.command(f"fix cold substrate langevin {equil_T} {equil_T} {damp} {seed + 31} zero yes")
    lmp.command("fix mom all momentum 100 linear 1 1 1")
    lmp.command(f"run {settings['melt_steps']}")
    if surface_wall.get("enabled"):
        lmp.command(f"unfix {surface_wall['fix_id']}")
    lmp.command("unfix hot")
    lmp.command("unfix cold")
    lmp.command(f"fix bath all langevin {equil_T} {equil_T} {damp} {seed + 47} zero yes")
    lmp.command(f"run {settings['equil_steps']}")
    lmp.command("write_data amorphous_300K.lmp")
    lmp.close()

    _write_metadata(work_dir, {
        "case": case_name,
        "state": "amorphous",
        "seed": seed,
        "output_structure": str(work_dir / "amorphous_300K.lmp"),
        "settings": settings,
        "melt_surface_wall": surface_wall,
    })
    return str(work_dir / "amorphous_300K.lmp")


@pipe.chore(num_tasks=TASKS_PER_CHORE, cores_per_task=CORES_PER_TASK, gpus_per_task=GPUS_PER_TASK)
def equilibrate_crystal(case_name: str, seed: int):
    case = _case_by_name(case_name)
    settings = _defaults(case)
    model_path = _abspath(CONFIG["model"])
    work_dir = ROOT / CONFIG["output_dir"] / case_name / "crystal" / f"seed_{seed}"

    lmp = _init_lammps(work_dir)
    _base_system(lmp, case, model_path)
    _add_dump(lmp, "crystal_equil.lammpstrj", settings["dump_every"])

    equil_T = settings["equil_temperature_K"]
    damp = settings["langevin_damp_ps"]
    lmp.command(f"velocity all create {equil_T} {seed} mom yes rot yes dist gaussian")
    lmp.command("fix int all nve")
    lmp.command(f"fix bath all langevin {equil_T} {equil_T} {damp} {seed + 71} zero yes")
    lmp.command("fix mom all momentum 100 linear 1 1 1")
    lmp.command(f"run {settings['equil_steps']}")
    lmp.command("write_data crystal_300K.lmp")
    lmp.close()

    _write_metadata(work_dir, {
        "case": case_name,
        "state": "crystal",
        "seed": seed,
        "output_structure": str(work_dir / "crystal_300K.lmp"),
        "settings": settings,
    })
    return str(work_dir / "crystal_300K.lmp")


@pipe.chore(num_tasks=TASKS_PER_CHORE, cores_per_task=CORES_PER_TASK, gpus_per_task=GPUS_PER_TASK)
def calibrate_springs(case_name: str, state: str, seed: int, prepared_structure: str = ""):
    case = _case_by_name(case_name)
    settings = _defaults(case)
    model_path = _abspath(CONFIG["model"])
    source = Path(prepared_structure) if prepared_structure else _state_source(case_name, state, seed)

    if not source.exists():
        raise FileNotFoundError(
            f"Spring calibration needs prepared structure first: {source}. "
            "Run amorphous/crystal preparation before --stages springs."
        )

    work_dir = ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "spring_calibration"
    lmp = _init_lammps(work_dir)
    spring_case = dict(case)
    spring_case["structure"] = str(source)
    spring_case["settings"] = dict(settings)
    spring_case["settings"]["replicate"] = [1, 1, 1]
    _base_system(lmp, spring_case, model_path)

    values = []
    for type_id, species in enumerate(case["species"], start=1):
        group = f"type_{type_id}"
        compute = f"msd_{type_id}"
        var = f"msd_{type_id}"
        lmp.command(f"group {group} type {type_id}")
        lmp.command(f"compute {compute} {group} msd com yes")
        lmp.command(f"variable {var} equal c_{compute}[4]")
        values.append(f"v_{var}")

    equil_T = settings["equil_temperature_K"]
    damp = settings["langevin_damp_ps"]
    lmp.command(f"velocity all create {equil_T} {seed + 101} mom yes rot yes dist gaussian")
    lmp.command("fix int all nve")
    lmp.command(f"fix bath all langevin {equil_T} {equil_T} {damp} {seed + 131} zero yes")
    lmp.command(f"fix msdout all ave/time {settings['msd_sample_every']} 1 {settings['msd_sample_every']} {' '.join(values)} file msd_species.dat")
    lmp.command(f"run {settings['msd_steps']}")
    lmp.close()

    if _rank0():
        msd_path = work_dir / "msd_species.dat"
        rows = []
        with msd_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= len(case["species"]) + 1:
                    rows.append([float(x) for x in parts[1:1 + len(case["species"])]])
        tail = rows[len(rows) // 2:] if rows else []
        mean_msd = []
        springs = {}
        for i, species in enumerate(case["species"]):
            values_i = [row[i] for row in tail if row[i] > 0.0]
            avg = sum(values_i) / len(values_i) if values_i else 0.0
            mean_msd.append(avg)
            springs[species] = (3.0 * KB_EV_PER_K * equil_T / avg) if avg > 0.0 else None

        with (work_dir / "spring_constants.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "case": case_name,
                "state": state,
                "seed": seed,
                "temperature_K": equil_T,
                "units": {
                    "msd": "angstrom^2",
                    "spring_constant": "eV/angstrom^2"
                },
                "species": case["species"],
                "mean_msd_by_species": dict(zip(case["species"], mean_msd)),
                "spring_constants_by_species": springs,
                "method": "K = 3 k_B T / <u^2>, using the second half of msd_species.dat"
            }, handle, indent=2)

    return str(work_dir / "spring_constants.json")


@pipe.chore(num_tasks=TASKS_PER_CHORE, cores_per_task=CORES_PER_TASK, gpus_per_task=GPUS_PER_TASK)
def run_frenkel_ladd(case_name: str, state: str, seed: int, prepared_structure: str = "", spring_constants: str = ""):
    case = _case_by_name(case_name)
    settings = _defaults(case)
    model_path = _abspath(CONFIG["model"])
    source = Path(prepared_structure) if prepared_structure else _state_source(case_name, state, seed)

    spring_path = Path(spring_constants) if spring_constants else (
        ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "spring_calibration" / "spring_constants.json"
    )
    if not source.exists():
        raise FileNotFoundError(f"Frenkel-Ladd needs prepared structure first: {source}")
    if not spring_path.exists():
        raise FileNotFoundError(f"Frenkel-Ladd needs spring calibration first: {spring_path}")

    springs = _load_spring_constants(spring_path, case["species"])
    work_dir = ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "frenkel_ladd"
    lmp = _init_lammps(work_dir)
    fl_case = dict(case)
    fl_case["structure"] = str(source)
    fl_case["settings"] = dict(settings)
    fl_case["settings"]["replicate"] = [1, 1, 1]
    _base_system(lmp, fl_case, model_path)

    fl_fixes = []
    for type_id, species in enumerate(case["species"], start=1):
        group = f"type_{type_id}"
        fix_id = f"fl_{type_id}"
        lmp.command(f"group {group} type {type_id}")
        lmp.command(
            f"fix {fix_id} {group} ti/spring {springs[species]} "
            f"{settings['fl_switch_steps']} {settings['fl_equil_steps']} "
            f"function {settings['fl_function']}"
        )
        fl_fixes.append(fix_id)

    harm_expr = "+".join(f"f_{fix_id}" for fix_id in fl_fixes)
    lmp.command("fix int all nve")
    lmp.command(
        f"fix bath all langevin {settings['equil_temperature_K']} "
        f"{settings['equil_temperature_K']} {settings['langevin_damp_ps']} {seed + 211} zero yes"
    )
    lmp.command(f"variable harm equal {harm_expr}")
    lmp.command("variable dE equal pe-v_harm")
    lmp.command(f"variable lambda equal f_{fl_fixes[0]}[1]")
    lmp.command("thermo_style custom step temp pe v_harm v_dE v_lambda")
    lmp.command(f"velocity all create {settings['equil_temperature_K']} {seed + 191} mom yes rot yes dist gaussian")
    lmp.command("velocity all zero linear")
    lmp.command(f"run {settings['fl_equil_steps']}")
    lmp.command('fix flprint all print 1 "${dE} ${lambda}" title "# dE [eV] lambda" screen no file forward.dat')
    lmp.command(f"run {settings['fl_switch_steps']}")
    lmp.command("unfix flprint")
    lmp.command(f"run {settings['fl_equil_steps']}")
    lmp.command('fix flprint all print 1 "${dE} ${lambda}" title "# dE [eV] lambda" screen no file backward.dat')
    lmp.command(f"run {settings['fl_switch_steps']}")
    lmp.command("unfix flprint")
    lmp.command("write_data fl_final.lmp")
    lmp.close()

    if _rank0():
        _postprocess_frenkel_ladd(work_dir, source, case, springs, settings["equil_temperature_K"], settings)
    return str(work_dir / "frenkel_ladd_result.json")


@pipe.chore(num_tasks=TASKS_PER_CHORE, cores_per_task=CORES_PER_TASK, gpus_per_task=GPUS_PER_TASK)
def run_reversible_scaling(case_name: str, state: str, seed: int, prepared_structure: str = "", fl_reference: str = ""):
    case = _case_by_name(case_name)
    settings = _defaults(case)
    model_path = _abspath(CONFIG["model"])
    source = Path(prepared_structure) if prepared_structure else _state_source(case_name, state, seed)
    fl_path = Path(fl_reference) if fl_reference else (
        ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / "frenkel_ladd" / "frenkel_ladd_result.json"
    )
    if not source.exists():
        raise FileNotFoundError(f"Reversible scaling needs prepared structure first: {source}")
    if not fl_path.exists():
        raise FileNotFoundError(f"Reversible scaling needs the FL reference first: {fl_path}")

    rs_dirname = settings.get("rs_output_dirname", "reversible_scaling")
    work_dir = ROOT / CONFIG["output_dir"] / case_name / state / f"seed_{seed}" / rs_dirname
    scale_method = settings.get("rs_scale_method", "hybrid_scaled")
    if scale_method not in ("fix_adapt", "force_scale", "hybrid_scaled"):
        raise ValueError(f"rs_scale_method must be fix_adapt, force_scale, or hybrid_scaled, got {scale_method!r}")
    use_kokkos = bool(settings.get("rs_use_kokkos", scale_method != "hybrid_scaled"))
    lmp = _init_lammps(work_dir, use_kokkos=use_kokkos)
    lmp.command("variable rs_lambda equal 1.0")
    rs_case = dict(case)
    rs_case["structure"] = str(source)
    rs_case["settings"] = dict(settings)
    rs_case["settings"]["replicate"] = [1, 1, 1]
    _base_system(
        lmp,
        rs_case,
        model_path,
        pair_scale_variable="rs_lambda" if scale_method == "hybrid_scaled" else None,
        use_kokkos=use_kokkos,
    )
    _add_dump(lmp, "rs.lammpstrj", settings["dump_every"])

    t_ref = float(settings["equil_temperature_K"])
    t_high = float(settings.get("rs_high_temperature_K", settings.get("melt_temperature_K", t_ref)))
    equil_steps = int(settings.get("rs_equil_steps", settings.get("fl_equil_steps", 5000)))
    high_equil_steps = int(settings.get("rs_high_equil_steps", equil_steps))
    switch_steps = int(settings.get("rs_switch_steps", settings.get("fl_switch_steps", 20000)))
    sample_every = int(settings.get("rs_sample_every", 1))
    damp = settings["langevin_damp_ps"]
    adapt_pstyle = settings.get("rs_fix_adapt_pair_style", "mliap")
    adapt_param = settings.get("rs_fix_adapt_parameter", "scale")
    lambda_span = t_high / t_ref - 1.0
    scaled_pe_expr = "$(v_rs_lambda*pe/atoms)" if scale_method == "force_scale" else "$(pe/atoms)"

    lmp.command(f"velocity all create {t_ref} {seed + 301} mom yes rot yes dist gaussian")
    lmp.command("fix int all nve")
    lmp.command("fix mom all momentum 100 linear 1 1 1")
    lmp.command("thermo_style custom step temp pe v_rs_lambda")

    lmp.command(f"fix bath all langevin {t_ref} {t_ref} {damp} {seed + 311} zero yes")
    lmp.command(f"run {equil_steps}")
    lmp.command("unfix bath")

    lmp.command(f"variable rs_lambda equal 1.0/(1.0+(elapsed/{switch_steps})*{lambda_span:.16g})")
    if scale_method == "fix_adapt":
        lmp.command(f"fix rsscale all adapt 1 pair {adapt_pstyle} {adapt_param} * * v_rs_lambda")
    elif scale_method == "force_scale":
        lmp.command("fix rsstore all store/force")
        lmp.command("variable rs_fx atom v_rs_lambda*f_rsstore[1]")
        lmp.command("variable rs_fy atom v_rs_lambda*f_rsstore[2]")
        lmp.command("variable rs_fz atom v_rs_lambda*f_rsstore[3]")
        lmp.command("fix rsscale all setforce v_rs_fx v_rs_fy v_rs_fz")
    lmp.command(f"fix bath all langevin {t_ref} {t_ref} {damp} {seed + 313} zero yes")
    lmp.command('print "# scaled_pe[eV/atom] lambda" file rs_forward.dat screen no')
    lmp.command(
        f'fix rsprint all print {sample_every} '
        f'"{scaled_pe_expr} ${{rs_lambda}}" '
        "screen no append rs_forward.dat"
    )
    lmp.command(f"run {switch_steps}")
    lmp.command("unfix rsprint")
    lmp.command("unfix bath")
    if scale_method == "fix_adapt":
        lmp.command("unfix rsscale")
    elif scale_method == "force_scale":
        lmp.command("unfix rsscale")
        lmp.command("unfix rsstore")
        lmp.command("variable rs_fx delete")
        lmp.command("variable rs_fy delete")
        lmp.command("variable rs_fz delete")
    lmp.command(f"variable rs_lambda equal {t_ref}/{t_high}")

    if scale_method == "force_scale":
        lmp.command("fix rsstore all store/force")
        lmp.command("variable rs_fx atom v_rs_lambda*f_rsstore[1]")
        lmp.command("variable rs_fy atom v_rs_lambda*f_rsstore[2]")
        lmp.command("variable rs_fz atom v_rs_lambda*f_rsstore[3]")
        lmp.command("fix rsscale all setforce v_rs_fx v_rs_fy v_rs_fz")
    lmp.command(f"fix bath all langevin {t_ref} {t_ref} {damp} {seed + 317} zero yes")
    lmp.command(f"run {high_equil_steps}")
    lmp.command("unfix bath")
    if scale_method == "force_scale":
        lmp.command("unfix rsscale")
        lmp.command("unfix rsstore")
        lmp.command("variable rs_fx delete")
        lmp.command("variable rs_fy delete")
        lmp.command("variable rs_fz delete")

    lmp.command(f"variable rs_lambda equal 1.0/(1.0+(1.0-(elapsed/{switch_steps}))*{lambda_span:.16g})")
    if scale_method == "fix_adapt":
        lmp.command(f"fix rsscale all adapt 1 pair {adapt_pstyle} {adapt_param} * * v_rs_lambda")
    elif scale_method == "force_scale":
        lmp.command("fix rsstore all store/force")
        lmp.command("variable rs_fx atom v_rs_lambda*f_rsstore[1]")
        lmp.command("variable rs_fy atom v_rs_lambda*f_rsstore[2]")
        lmp.command("variable rs_fz atom v_rs_lambda*f_rsstore[3]")
        lmp.command("fix rsscale all setforce v_rs_fx v_rs_fy v_rs_fz")
    lmp.command(f"fix bath all langevin {t_ref} {t_ref} {damp} {seed + 319} zero yes")
    lmp.command('print "# scaled_pe[eV/atom] lambda" file rs_backward.dat screen no')
    lmp.command(
        f'fix rsprint all print {sample_every} '
        f'"{scaled_pe_expr} ${{rs_lambda}}" '
        "screen no append rs_backward.dat"
    )
    lmp.command(f"run {switch_steps}")
    lmp.command("unfix rsprint")
    lmp.command("unfix bath")
    if scale_method == "fix_adapt":
        lmp.command("unfix rsscale")
    elif scale_method == "force_scale":
        lmp.command("unfix rsscale")
        lmp.command("unfix rsstore")
        lmp.command("variable rs_fx delete")
        lmp.command("variable rs_fy delete")
        lmp.command("variable rs_fz delete")
    lmp.command("write_data rs_final.lmp")
    lmp.close()

    if _rank0():
        _postprocess_reversible_scaling(work_dir, source, case, state, seed, settings, fl_path)
    return str(work_dir / "rs_free_energy_result.json")


def _selected_cases(names):
    if not names:
        return [case["name"] for case in CONFIG["cases"]]
    known = {case["name"] for case in CONFIG["cases"]}
    for name in names:
        if name not in known:
            raise ValueError(f"Unknown case {name}. Known cases: {sorted(known)}")
    return names


def _stage_output(case_name, stage, seed, state=None):
    seed_dir = ROOT / CONFIG["output_dir"] / case_name
    if stage == "amorphous":
        return seed_dir / "amorphous" / f"seed_{seed}" / "amorphous_300K.lmp"
    if stage == "crystal":
        return seed_dir / "crystal" / f"seed_{seed}" / "crystal_300K.lmp"
    if stage == "springs":
        return seed_dir / state / f"seed_{seed}" / "spring_calibration" / "spring_constants.json"
    if stage == "frenkel-ladd":
        return seed_dir / state / f"seed_{seed}" / "frenkel_ladd" / "frenkel_ladd_result.json"
    if stage in ("reversible-scaling", "rs"):
        settings = _defaults(_case_by_name(case_name))
        rs_dirname = settings.get("rs_output_dirname", "reversible_scaling")
        return seed_dir / state / f"seed_{seed}" / rs_dirname / "rs_free_energy_result.json"
    raise ValueError(f"Unknown stage: {stage}")


def _maybe_skip(args, case_name, stage, seed, state=None):
    if not args.skip_existing:
        return False
    output = _stage_output(case_name, stage, seed, state)
    if output.exists():
        state_label = f"/{state}" if state else ""
        print(f"skip-existing: {case_name}{state_label} seed {seed} {stage}: {output}", flush=True)
        return True
    return False


def _state_seeds(args, state, default_seeds):
    if state == "amorphous" and args.amorphous_seeds is not None:
        return args.amorphous_seeds
    if state == "crystal" and args.crystal_seeds is not None:
        return args.crystal_seeds
    return default_seeds


def _queue_prepare(args, case_name, state, seed):
    if state == "amorphous":
        if _maybe_skip(args, case_name, "amorphous", seed):
            return str(_stage_output(case_name, "amorphous", seed))
        return prepare_amorphous(case_name=case_name, seed=seed)
    if state == "crystal":
        if _maybe_skip(args, case_name, "crystal", seed):
            return str(_stage_output(case_name, "crystal", seed))
        return equilibrate_crystal(case_name=case_name, seed=seed)
    raise ValueError(f"Unknown state: {state}")


def _queue_dependency_chain(args, case_name, state, seed):
    prepared = _queue_prepare(args, case_name, state, seed)

    if _maybe_skip(args, case_name, "springs", seed, state=state):
        springs = str(_stage_output(case_name, "springs", seed, state=state))
    else:
        springs = calibrate_springs(case_name, state, seed, prepared)

    if _maybe_skip(args, case_name, "frenkel-ladd", seed, state=state):
        fl_reference = str(_stage_output(case_name, "frenkel-ladd", seed, state=state))
    else:
        fl_reference = run_frenkel_ladd(case_name, state, seed, prepared, springs)

    rs_stage = "rs" if "rs" in args.stages else "reversible-scaling"
    if _maybe_skip(args, case_name, rs_stage, seed, state=state):
        return str(_stage_output(case_name, rs_stage, seed, state=state))
    return run_reversible_scaling(case_name, state, seed, prepared, fl_reference)


def main():
    parser = argparse.ArgumentParser(description="MatEnsemble MACE recrystallization workflow for heterostructures.")
    parser.add_argument("--workflow", choices=["stage", "full-dag"], default="full-dag",
                        help="stage preserves the legacy stage queue; full-dag wires prep->springs->FL->RS dependencies in one MatEnsemble workflow.")
    parser.add_argument("--stages", nargs="+", default=["amorphous", "crystal"],
                        choices=["amorphous", "crystal", "springs", "frenkel-ladd", "reversible-scaling", "rs"],
                        help="Workflow stages to queue.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case names from recryst_config.json.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Seed list. Defaults to config seeds.")
    parser.add_argument("--amorphous-seeds", nargs="*", type=int, default=None,
                        help="Seed list for amorphous stages; overrides --seeds for amorphous states.")
    parser.add_argument("--crystal-seeds", nargs="*", type=int, default=None,
                        help="Seed list for crystal stages; overrides --seeds for crystal states.")
    parser.add_argument("--spring-states", nargs="+", default=["amorphous", "crystal"],
                        choices=["amorphous", "crystal"],
                        help="States for spring, Frenkel-Ladd, and reversible-scaling stages.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not queue a chore when its final output file already exists.")
    args = parser.parse_args()

    cases = _selected_cases(args.cases)
    seeds = args.seeds if args.seeds else CONFIG.get("seeds", [101])

    if args.workflow == "full-dag":
        for case_name in cases:
            for seed in _state_seeds(args, "amorphous", seeds):
                if "amorphous" in args.spring_states:
                    _queue_dependency_chain(args, case_name, "amorphous", seed)
            for seed in _state_seeds(args, "crystal", seeds):
                if "crystal" in args.spring_states:
                    _queue_dependency_chain(args, case_name, "crystal", seed)
        pipe.submit()
        return

    for case_name in cases:
        for seed in _state_seeds(args, "amorphous", seeds):
            if "amorphous" in args.stages and not _maybe_skip(args, case_name, "amorphous", seed):
                prepare_amorphous(case_name=case_name, seed=seed)
        for seed in _state_seeds(args, "crystal", seeds):
            if "crystal" in args.stages and not _maybe_skip(args, case_name, "crystal", seed):
                equilibrate_crystal(case_name=case_name, seed=seed)
        if "springs" in args.stages:
            for state in args.spring_states:
                for seed in _state_seeds(args, state, seeds):
                    if not _maybe_skip(args, case_name, "springs", seed, state=state):
                        calibrate_springs(case_name=case_name, state=state, seed=seed)
        if "frenkel-ladd" in args.stages:
            for state in args.spring_states:
                for seed in _state_seeds(args, state, seeds):
                    if not _maybe_skip(args, case_name, "frenkel-ladd", seed, state=state):
                        run_frenkel_ladd(case_name=case_name, state=state, seed=seed)
        if "reversible-scaling" in args.stages or "rs" in args.stages:
            rs_stage = "rs" if "rs" in args.stages else "reversible-scaling"
            for state in args.spring_states:
                for seed in _state_seeds(args, state, seeds):
                    if not _maybe_skip(args, case_name, rs_stage, seed, state=state):
                        run_reversible_scaling(case_name=case_name, state=state, seed=seed)

    pipe.submit()


if __name__ == "__main__":
    main()
