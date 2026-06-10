import types

# h5py in this container was compiled against HDF5 2.0.0 but bundles 1.14.3,
# causing a crash on import. mace.data.utils imports it at module level but
# only uses it for training data (HDF5 files), not inference. Stub it out.
_h5py_stub = types.ModuleType("h5py")
import sys

sys.modules["h5py"] = _h5py_stub

import lammps
import lammps.mliap

# Patch MACEEdgeForcesWrapper: newer MACE code expects self.total_charge but
# older model files don't include it.
from mace.calculators.lammps_mliap_mace import MACEEdgeForcesWrapper as _MACEWrapper
from matensemble.pipeline import Pipeline

# Default to 0 for neutral systems.
if not hasattr(_MACEWrapper, "total_charge"):
    _MACEWrapper.total_charge = 0.0

pipe = Pipeline()


@pipe.chore(gpus_per_task=1)
def get_forces_lmp(structure: str, ffield: str):
    """
    Run a LAMMPS minimization and return atomic forces.

    Args:
        structure: Path to the LAMMPS data file (structure).
        ffield:    Path to the MLIAP/MACE force field file.

    Returns:
        numpy array of forces on each atom.
    """

    import sys, types

    if "h5py" not in sys.modules:
        sys.modules["h5py"] = types.ModuleType("h5py")

    lmp = lammps.lammps(
        cmdargs=[
            "-k",
            "on",
            "g",
            "1",
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
dimension     3
boundary      p p p
"""
    lmp.commands_string(init_lmp)
    lmp.command(f"read_data {structure}")
    lmp.command(f"pair_style mliap unified {ffield} 0")
    lmp.command("pair_coeff * * C Cu O")
    lmp.command("thermo 10")
    lmp.command("minimize 0 1e-5 1000 1000")
    lmp.command("write_data minimized_structure.lmp")

    forces = lmp.numpy.extract_atom("f")
    print("forces:", forces)
    return forces


# --- Example usage ---
if __name__ == "__main__":
    # /pscratch/sd/k/kaleb/neil_tests/new_image/LAMMPS/GPU/MACE/Simulation
    forces = get_forces_lmp(
        structure="/pscratch/sd/k/kaleb/neil_tests/new_image/LAMMPS/GPU/MACE/Simulation/CO_art_str.lmp",
        ffield="/pscratch/sd/k/kaleb/neil_tests/new_image/LAMMPS/GPU/MACE/Simulation/mace-omat-0-medium.model-mliap_lammps.pt",
    )

    pipe.submit()
