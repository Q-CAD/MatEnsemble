import numpy as np
import glob
import os
import sys
import argparse
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.lammpsrun import read_lammps_dump_text
from pathlib import Path
import warnings


def get_locations(path, ffield_name):
    run_paths = []
    for root, _, _ in os.walk(path):
        ffield = glob.glob(os.path.join(root, ffield_name))
        if ffield:
            run_paths.append(os.path.abspath(root))
    return run_paths


def get_structure_paths_and_tasks(run_paths, in_directory, structure, atoms_per_task):
    if (
        ".lmp" in structure
    ):  # Structure comes from LAMMPs input file in the input directory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure_paths = [
                f"{os.path.abspath(os.path.join(in_directory, structure))}"
                for path in run_paths
            ]
            structures = [
                LammpsData.from_file(structure_path, atom_style="charge").structure
                for structure_path in structure_paths
            ]
            c_run_paths = run_paths

    elif (
        ".res" in structure
    ):  # Structure comes from a restart file in the run directory
        # THIS STILL NEEDS TO BE TESTED
        unchecked_structure_paths = [
            f"{os.path.join(path, structure)}" for path in run_paths
        ]
        existing_structure_paths = [
            p for p in unchecked_structure_paths if os.path.exists(p)
        ]

        aaa = AseAtomsAdaptor()
        c_run_paths, structure_paths, structures = [], [], []
        for i, path in enumerate(existing_structure_paths):
            dumpfiles = sorted(list(Path(path).parent.glob("*.dump")))
            if dumpfiles:
                atoms = read_lammps_dump_text(open(dumpfiles[-1], "r"))
                structures.append(aaa.get_structure(atoms))
                structure_paths.append(path)
                c_run_paths.append(run_paths[i])

    else:
        raise ValueError(f"Unsupported structure file {structure}")

    return (
        c_run_paths,
        structure_paths,
        [np.floor(len(s) / atoms_per_task).astype(int) for s in structures],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Argument parser to run LAMMPs with Flux using template files"
    )

    # Add arguments

    # Input and ouput paths, files and names
    parser.add_argument(
        "--ffields_directory",
        "-ffd",
        help="Path to the directory tree with force field files",
        default=".",
    )
    parser.add_argument(
        "--ffield_name", "-ffn", help="Name of force field files", default="ffield"
    )
    parser.add_argument(
        "--task_script",
        "-ts",
        help="Name of the task Python script",
        default="lammps_flux_kernel.py",
    )

    parser.add_argument(
        "--in_directory",
        "-id",
        help="Name of the directory to check for input files",
        default="flux_inputs",
    )
    parser.add_argument(
        "--in_name", "-in", help="Name of the task Python script", default="in.flux"
    )
    parser.add_argument(
        "--structure",
        "-s",
        help="Name of the .lmp file OR .res file",
        default="structure.lmp",
    )  # Build out for continuation jobs here
    parser.add_argument(
        "--control_name",
        "-cn",
        help="Name of the ReaxFF control file",
        default="control.reax_c.rdx",
    )  # Build out for contiuation jobs here

    parser.add_argument(
        "--atoms_per_task", "-apt", help="Atoms per task", type=int, default=90
    )
    parser.add_argument(
        "--dry_run",
        "-dry",
        help="Only print the structures to be run",
        action="store_true",
    )

    args = parser.parse_args()

    task_command = "".join(["python", " ", str(Path.absolute(Path(args.task_script)))])
    run_paths = get_locations(os.path.abspath(args.ffields_directory), args.ffield_name)
    c_run_paths, structure_paths, run_tasks = get_structure_paths_and_tasks(
        run_paths, args.in_directory, args.structure, args.atoms_per_task
    )

    task_list = [i for i in range(len(c_run_paths))]
    task_arg_list = [
        [
            f"{os.path.abspath(os.path.join(args.in_directory, args.in_name))}",  # in file template location
            f"{os.path.abspath(structure_paths[i])}",  # structure file location
            f"{os.path.abspath(os.path.join(c_run_paths[i], args.ffield_name))}",  # ffield file location
            f"{os.path.abspath(os.path.join(args.in_directory, args.control_name))}",
        ]
        for i, run_path in enumerate(c_run_paths)
    ]

    if args.dry_run:
        for i, task_arg in enumerate(task_arg_list):
            print(f"path: {structure_paths[i]}, cores: {run_tasks[i]}\n")
        print(f"Total cores required = {np.sum(run_tasks)}")

    else:
        from matensemble.manager import SuperFluxManager

        master = SuperFluxManager(
            task_list,
            task_command,
            None,
            tasks_per_job=run_tasks,
            cores_per_task=1,
            gpus_per_task=0,
            write_restart_freq=5,
        )
        master.poolexecutor(
            task_arg_list=task_arg_list, buffer_time=1, task_dir_list=c_run_paths
        )

        return


if __name__ == "__main__":
    main()
