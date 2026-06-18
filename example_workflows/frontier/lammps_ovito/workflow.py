"""
MatEnsemble campaign: frontier_copper_melt_ovito

Science goal:
Run a copper melt simulation on Frontier with LAMMPS using the Cu_u3.eam potential,
then render a final OVITO visualization from the trajectory.
"""

import os
from pathlib import Path

from lammps import lammps
from ovito.io import import_file
from ovito.vis import TachyonRenderer, Viewport

from matensemble.pipeline import Pipeline

pipe = Pipeline()
WORKFLOW_DIR = Path(__file__).resolve().parent


def locate_cu_potential() -> Path:
    """Find the Cu_u3.eam potential in the current workspace or common install locations."""

    potential_candidates = []
    potential_candidates.extend(
        [
            WORKFLOW_DIR / "Cu_u3.eam",
            WORKFLOW_DIR / "potentials" / "Cu_u3.eam",
        ]
    )
    potentials_root = os.environ.get("LAMMPS_POTENTIALS")
    if potentials_root:
        root_path = Path(potentials_root)
        potential_candidates.extend(
            [
                root_path / "Cu_u3.eam",
                root_path / "potentials" / "Cu_u3.eam",
            ]
        )

    potential_candidates.extend(
        [
            Path("Cu_u3.eam"),
            Path("potentials") / "Cu_u3.eam",
            Path("/usr/share/lammps/potentials/Cu_u3.eam"),
            Path("/usr/local/share/lammps/potentials/Cu_u3.eam"),
            Path("/opt/share/lammps/potentials/Cu_u3.eam"),
        ]
    )

    for candidate in potential_candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"- {candidate}" for candidate in potential_candidates)
    raise FileNotFoundError(
        "Could not find Cu_u3.eam. Looked in these locations:\n"
        f"{searched}\n"
        "Place the file in the workflow directory or set LAMMPS_POTENTIALS."
    )


@pipe.chore(name="simulate_copper_melt", num_tasks=1, cores_per_task=1, gpus_per_task=1)
def simulate_copper_melt() -> dict:
    """Run a small copper heating/melting trajectory and write a LAMMPS dump."""

    run_dir = Path("copper_melt_run")
    dump_dir = run_dir / "dumps"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    potential_path = locate_cu_potential()
    dump_path = dump_dir / "copper_melt.lammpstrj"
    data_path = run_dir / "copper_melt.data"
    log_path = run_dir / "lammps.log"

    cmdargs = [
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-pk",
        "kokkos",
        "-screen",
        "none",
    ]
    lmp = lammps(cmdargs=cmdargs)

    lmp.command(f"log {log_path}")
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    lmp.command("lattice fcc 3.615")
    lmp.command("region box block 0 8 0 8 0 8")
    lmp.command("create_box 1 box")
    lmp.command("create_atoms 1 box")
    lmp.command("mass 1 63.546")
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify delay 5 every 1 check yes")
    lmp.command("velocity all create 300.0 4928459 mom yes rot yes dist gaussian")
    lmp.command("pair_style eam")
    lmp.command(f"pair_coeff * * {potential_path}")
    lmp.command("timestep 0.001")
    lmp.command("thermo 200")
    lmp.command("thermo_style custom step temp pe ke etotal press")
    lmp.command(f"dump melt all custom 200 {dump_path} id type x y z")
    lmp.command("dump_modify melt sort id")
    lmp.command("fix melt all nvt temp 300.0 1800.0 0.1")
    lmp.command("run 4000")
    lmp.command("unfix melt")
    lmp.command("fix hold all nvt temp 1800.0 1800.0 0.1")
    lmp.command("run 1000")
    lmp.command(f"write_data {data_path}")

    final_temperature = lmp.get_thermo("temp")
    final_potential_energy = lmp.get_thermo("pe")

    print(f"LAMMPS dump written to {dump_path}")
    print(f"LAMMPS data file written to {data_path}")

    return {
        "dump_path": str(dump_path),
        "data_path": str(data_path),
        "log_path": str(log_path),
        "final_temperature": final_temperature,
        "final_potential_energy": final_potential_energy,
    }


@pipe.chore(name="render_copper_visual")
def render_copper_visual(simulation_result: dict) -> dict:
    """Render the last trajectory frame with OVITO and save a PNG image."""

    dump_path = Path(simulation_result["dump_path"])
    render_dir = dump_path.parent.parent / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    image_path = render_dir / "copper_melt_ovito.png"

    pipeline = import_file(str(dump_path))
    pipeline.add_to_scene()

    viewport = Viewport(type=Viewport.Type.Perspective)
    viewport.zoom_all()
    viewport.render_image(
        filename=str(image_path),
        size=(1400, 900),
        frame=pipeline.source.num_frames - 1,
        renderer=TachyonRenderer(),
    )

    pipeline.remove_from_scene()

    print(f"OVITO render written to {image_path}")

    return {
        "image_path": str(image_path),
        "source_dump": str(dump_path),
    }


simulation = simulate_copper_melt()
render = render_copper_visual(simulation)

future = pipe.submit(log_delay=5, dashboard=False)
print(future.result())
