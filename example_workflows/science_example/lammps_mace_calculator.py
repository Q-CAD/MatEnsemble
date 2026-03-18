import numpy as np
import lammps
import lammps.mliap

from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.job(
    name="lammps-mace-calc",
    num_tasks=1,
    cores_per_task=1,
    gpus_per_task=4,
    mpi=False,
)
def run_lammps_mace(output_file: str, ff_file: str):
    lmp = lammps.lammps(
        cmdargs=[
            "-k",
            "on",
            "g",
            "4",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "neigh",
            "half",
            "newton",
            "off",
            "-echo",
            "both",
            "-log",
            "none",
        ]
    )

    lammps.mliap.activate_mliappy_kokkos(lmp)

    init_lmp = """
    units         metal
    atom_style    atomic
    atom_modify   map yes
    newton        on
    processors    * * *
    dimension 3
    boundary p p p
    """

    lmp.commands_string(init_lmp)
    lmp.command(f"read_data {output_file}")
    lmp.command(f"pair_style mliap unified {ff_file} 0")
    lmp.command("pair_coeff * * C Cu O")

    lmp.command("thermo 10")
    lmp.command("minimize 0 1e-5 1000 1000")
    lmp.command("write_data minimized_structure.lmp")

    f_raw = lmp.gather_atoms("f", 1, 3)
    force = np.array(f_raw).reshape(-1, 3)

    if lmp.comm.Get_rank() == 0:
        print("forces:", force, force.shape)
        return force
    else:
        return "Check force !"
