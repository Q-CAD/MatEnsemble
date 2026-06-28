import datetime

from pathlib import Path

from networkx import write_network_text

from matensemble.manager import FluxManager
from matensemble.chore import Chore, ChoreType
from matensemble.model import Resources

# from matensemble.dynopro.driver import online_dynamics


class EnsembleDynamicsRunner:
    """
    A class to manage the execution of multiple MD simulations in parallel using SuperFluxManager.

    Attributes:
        sim_list (list): List of simulation identifiers.
        sim_args_list (list): List of arguments for each simulation.
        task_dir_list (list): List of directories for each simulation.
        tasks_per_job (int): Number of tasks per job.
        cores_per_task (int): Number of cores per task.
        gpus_per_task (int): Number of GPUs per task.
        write_restart_freq (int | None): Reserved for future restart support.
        buffer_time (float): Buffer time for task execution.
    """

    def __init__(
        self,
        sim_list,
        sim_args_list,
        sim_dir_list,
        tasks_per_job=1,
        cores_per_task=1,
        gpus_per_task=0,
        write_restart_freq=None,
        buffer_time=0.1,
        adaptive=False,
        nnodes=None,
        gpus_per_node=None,
    ):
        self.sim_list = sim_list
        self.sim_args_list = sim_args_list
        self.sim_dir_list = sim_dir_list
        self.tasks_per_job = tasks_per_job
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task
        self.nnodes = nnodes
        self.gpus_per_node = gpus_per_node
        self.write_restart_freq = write_restart_freq
        self.buffer_time = buffer_time
        self.adaptive = adaptive
        self.sim_command = "python -m matensemble.dynopro.driver"

    def build_dynopro_chores(
        self,
        outdir: Path,
    ) -> list[Chore]:
        chores = []

        length = len(self.sim_list)
        if length != len(self.sim_args_list):
            raise Exception(
                "Dynopro Error: the length of the simulation list and simulation argument list are not equal."
            )
        else:
            for i in range(length):
                popped = self.sim_list.pop(0)
                chore_id = f"chore-dynopro-{popped}-{i:04d}"
                resources = Resources(
                    num_tasks=self.tasks_per_job,
                    cores_per_task=self.cores_per_task,
                    gpus_per_task=self.gpus_per_task,
                    mpi=True,
                )
                workdir = outdir / chore_id

                args = self.sim_args_list.pop(0)
                command = f"{self.sim_command} '{args}'"
                chores.append(
                    Chore(
                        id=chore_id,
                        command=command,
                        chore_type=ChoreType.EXECUTABLE,
                        resources=resources,
                        workdir=workdir,
                        nnodes=self.nnodes,
                    )
                )

        return chores

    def run(self):
        """
        Run the ensemble dynamics simulations.

        This method initializes the SuperFluxManager and executes the simulations in parallel.
        """

        root = Path.cwd()
        basedir = root / f"matensemble_workflow-{datetime.datetime.now():%Y%m%d_%H%M%S}"
        outdir = basedir / "out"

        chores = self.build_dynopro_chores(outdir)

        # Initialize FluxManager
        fm = FluxManager(
            chore_list=chores,
            base_dir=basedir,
            # write_restart_freq=self.write_restart_freq, # leave this out for now until restarts are ready for use
        )

        # Execute the simulations
        fm.run(
            buffer_time=self.buffer_time,
            adaptive=self.adaptive,
            dynopro=True,
        )
