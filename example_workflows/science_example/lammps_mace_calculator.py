import numpy as np
import lammps
import lammps.mliap
import os
import shutil
import ctypes
import pathlib
import subprocess

from matensemble.pipeline import Pipeline


pipe = Pipeline()


@pipe.chore(
    name="lammps-mace-calc",
    num_tasks=1,
    cores_per_task=1,
    gpus_per_task=4,
    mpi=True,
    inherit_env=True,
)
def run_lammps_mace(output_file: str, ff_file: str):
    print("OS ENVIRONMENT", os.environ)
    print("LAMMPS executable:", shutil.which("lmp"))
    print("PATH:", os.environ.get("PATH"))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("ROCR_VISIBLE_DEVICES:", os.environ.get("ROCR_VISIBLE_DEVICES"))

    pkgdir = pathlib.Path(lammps.__file__).resolve().parent
    print("LAMMPS package dir:", pkgdir)
    print("pkg-local lib exists:", (pkgdir / "liblammps.so").exists())

    for cand in [
        pkgdir / "liblammps.so",
        pathlib.Path("/opt/lammps/install/lib/liblammps.so"),
        pathlib.Path("/opt/lammps/build/liblammps.so"),
    ]:
        print(f"\nTrying CDLL on: {cand}")
        print("exists:", cand.exists())
        if cand.exists():
            try:
                ctypes.CDLL(str(cand))
                print("CDLL load: OK")
            except OSError as e:
                print("CDLL load failed:", e)
                subprocess.run(["/usr/bin/ldd", str(cand)], check=False)

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
