from lammps import lammps
from matensemble.pipeline import Pipeline


pipe = Pipeline()


@pipe.chore(num_tasks=1, cores_per_task=1)
def run_lammps_smoke():
    """Run a tiny CPU LAMMPS Lennard-Jones melt and return potential energy."""

    lmp = lammps(cmdargs=["-screen", "none"])
    lmp.command("units           lj")
    lmp.command("atom_style      atomic")
    lmp.command("lattice         fcc 0.8442")
    lmp.command("region          box block 0 8 0 8 0 8")
    lmp.command("create_box      1 box")
    lmp.command("create_atoms    1 box")
    lmp.command("mass            1 1.0")
    lmp.command("velocity        all create 1.44 87287")
    lmp.command("pair_style      lj/cut 2.5")
    lmp.command("pair_coeff      1 1 1.0 1.0 2.5")
    lmp.command("run             20")

    potential_energy = lmp.get_thermo("pe")
    lmp.close()
    return float(potential_energy)


energy = run_lammps_smoke()
future = pipe.submit(log_delay=1)
future.result()
print(energy.result())
