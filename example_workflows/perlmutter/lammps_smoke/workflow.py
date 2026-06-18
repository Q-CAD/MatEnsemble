from lammps import lammps
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.chore(num_tasks=1, gpus_per_task=1)
def run_hello_world_kokkos(num_gpus=1):
    """
    Runs the equivalent of a LAMMPS 'Hello World' (LJ atomic melt)
    accelerated on a GPU using the Kokkos package.
    """
    print(f"Initializing LAMMPS with Kokkos package using {num_gpus} GPU(s)...")

    # 1. Define command line arguments for Kokkos
    # Equivalent to command line flags: -k on g 1 -sf kk -pk kokkos
    cmdargs = [
        "-k",
        "on",
        "g",
        str(num_gpus),  # Enable Kokkos and request GPU(s)
        "-sf",
        "kk",  # Append the /kk suffix to supported styles
        "-pk",
        "kokkos",  # Modify default Kokkos settings
        "-screen",
        "none",  # Optional: Suppress screen output to terminal
    ]

    # 2. Initialize the LAMMPS python object with arguments
    lmp = lammps(cmdargs=cmdargs)

    # 3. Supply the Lennard-Jones input script line-by-line
    # (Note: styles like pair_style lj/cut automatically resolve to lj/cut/kk)
    lmp.command("units           lj")
    lmp.command("atom_style      atomic")
    lmp.command("lattice         fcc 0.8442")
    lmp.command("region          box block 0 20 0 20 0 20")
    lmp.command("create_box      1 box")
    lmp.command("create_atoms    1 box")
    lmp.command("mass            1 1.0")
    lmp.command("velocity        all create 1.44 87287")

    # Kokkos intercepts and offloads these interactions directly to the GPU
    lmp.command("pair_style      lj/cut 2.5")
    lmp.command("pair_coeff      1 1 1.0 1.0 2.5")

    # 4. Run the simulation steps
    print("Executing 100 simulation steps on GPU...")
    lmp.command("run             100")

    # 5. Extract a quick metric to prove success (e.g., total energy)
    thermo_energy = lmp.get_thermo("pe")
    print(f"Simulation completed successfully!")
    print(f"Final Potential Energy: {thermo_energy:.4f}")

    # 6. Cleanly close the LAMMPS instance
    lmp.close()


run_hello_world_kokkos()

pipe.submit(log_delay=1)

print(pipe.results())
