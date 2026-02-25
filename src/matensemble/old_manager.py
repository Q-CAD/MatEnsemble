"""
manager.py
----------

The manager holds the core logic of MatEnsemble.
It handles the four task lists (pending_tasks, running_tasks, completed_tasks,
failed_tasks), the resource count and the scheduling and processing of tasks
through the SuperFluxManager object.

The poolexecutor method can take optional arguments of type TaskSubmissionStrategy
and FutureProcessingStrategy. If these arguments are provided then the
scheduling loop will use these strategies to submit the tasks and process the tasks,
if they are not provided the strategies will be infered at run-time.

To see about implementing your own strategies look at the strategy/ sub-package

"""

from matensemble.pipeline.compile import TaskSpec
import numpy as np
import flux.job
import os.path
import logging
import numbers
import pickle
import time
import flux
import copy
import sys
import os

from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy
from matensemble.strategy.submission_strategy_base import TaskSubmissionStrategy
from matensemble.strategy.non_adaptive_strategy import NonAdaptiveStrategy
from matensemble.strategy.cpu_affine_strategy import CPUAffineStrategy
from matensemble.strategy.gpu_affine_strategy import GPUAffineStrategy
from matensemble.strategy.adaptive_strategy import AdaptiveStrategy
from matensemble.strategy.dynopro_strategy import DynoproStrategy
from matensemble.logger import setup_workflow_logging
from collections import deque
from datetime import datetime

__author__ = ["Soumendu Bagchi", "Kaleb Duchesneau"]
__package__ = "matensemble"


class Task:
    def __init__(
        self,
        id: int | str,
        subtasks: list[SubTask],
        cores_per_task: int,
        gpus_per_task: int,
        dependencies: list[int | str] | None = None,
    ) -> None:
        """
        task objects n
        """

        self.id = id
        self.command = command
        self.subtasks = subtasks
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task


class SubTask:
    def __init__(self, command: str) -> None:
        self.command = commmand


class SuperFluxManager:
    """
    Task submission manager that orchestrates high-throughput Flux jobs
    with 'Fluxlets' and tracks state and periodically 'pickles' restart files.

    Attributes
    ----------
    pending_tasks: deque
        A double ended-queue for tasks that are yet to start running or be
        completed
    running_tasks: set
        The tasks that are currently running
    completed_tasks: list
        Tasks that have been completed successfully
    failed_tasks: list
        Tasks that have failed
    flux_handle: flux.Flux()
        The flux handle for resource managment and task execution
    futures: set
        Set of FluxExecutorFuture objects representing the tasks output
    tasks_per_job: deque
        Double ended-queue of integers representing the number of sub-tasks per
        task
    cores_per_task: int
        The number of CPU cores that will be allocated to each task
    gpus_per_task: int
        The number of GPUs that will be allocated to each task
    gen_task_cmd: str
        The general command that each task will follow
    write_restart_frew: int
        How many tasks need to be completed before creating another restart file
    executor: flux.Executor()
        Executor for task submission
    logger: logging.Logger()
        Handles logging status and updates
    status: matensemble.logging.StatusWriter()
        Writes to a status file that the user can watch to see progress updates
    paths: matensemble.logger.WorkflowPaths
        Creates the output directories in the following structure::

            <cwd>/<SLURM_JOB_ID>_matensemble_workflow/
              ├── status.log
              ├── logs/
              │   └── <timestamp>_matensemble_workflow.log
              └── out/
                  └── <output_of_workflow>


    Notes
    -----
    buffer_time is used by strategies both as a polling timeout and as an intentional
    submission pacing delay (time.sleep), so large values can significantly slow
    end-to-end throughput.
    """

    def __init__(
        self,
        tasks: list[TaskSpec],
        write_restart_freq: int | None = 100,
        nnodes: int | None = None,
        gpus_per_node: int | None = None,
        restart_filename: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tasks: list[TaskSpec]
            The list of task specifications (either python functions or executable files)
            to submit to flux.
        write_restart_freq : int, optional
            Number of tasks that must complete before writing a pickled restart file.
        nnodes : int, optional
            Number of nodes allocated for the ensemble.
        gpus_per_node : int, optional
            Number of GPUs available per node.
        restart_filename : str, optional
            Path to a restart file. If provided and points to a ``.dat`` file, the
            ensemble resumes from the saved state.

        """

        self.running_tasks = set()
        self.completed_tasks = []
        self.failed_tasks = []
        self.futures = set()

        self.flux_handle = flux.Flux()

        self.nnodes = nnodes
        self.gpus_per_node = gpus_per_node
        self.write_restart_freq = write_restart_freq

        self.logger, self.status, self.paths = setup_workflow_logging()
        self.load_restart(restart_filename)

    def load_restart(self, filename: str | None = None) -> None:
        """
        Sets the completed_tasks, running_tasks, pending_tasks and failed_tasks
        and restarts the ensemble

        Parameters
        ----------
        filename: str
            The name of the file to restart from

        Return
        ------
        None
        """

        if (filename is not None) and os.path.isfile(filename):
            try:
                task_log = pickle.load(open(filename, "rb"))
                self.completed_tasks = task_log["Completed tasks"]
                self.running_tasks = task_log["Running tasks"]
                self.pending_tasks = task_log["Pending tasks"]
                self.failed_tasks = task_log["Failed tasks"]
                # self.logger.info(
                #     "================= WORKFLOW RESTARTING =================="
                # )
            except Exception as e:
                self.logger.warning("%s", e)
                raise e

    def create_restart_file(self) -> None:
        """
        Pickles the current state of the program into a file

        Return
        ------
        None
        """

        self.task_log = {
            "Completed tasks": self.completed_tasks,
            "Running tasks": self.running_tasks,
            "Pending tasks": self.pending_tasks,
            "Failed tasks": self.failed_tasks,
        }
        pickle.dump(
            self.task_log,
            open(f"restart_{len(self.completed_tasks)}.dat", "wb"),
        )

    def check_resources(self) -> None:
        """
        Gets the available resources from Flux's Remote Procedure Call (RPC)

        Return
        ------
        None
        """

        self.resource_status = flux.resource.status.ResourceStatusRPC(
            self.flux_handle
        ).get()
        self.resource_list = flux.resource.list.resource_list(self.flux_handle).get()
        self.resource = flux.resource.list.resource_list(self.flux_handle).get()
        self.free_gpus = self.resource.free.ngpus
        self.free_cores = self.resource.free.ncores
        self.free_excess_cores = self.free_cores - self.free_gpus

    def log_progress(self) -> None:
        """
        Logs the current state of the program, pending_tasks, running_tasks,
        completed_tasks, failed_tasks and a resource count free_cores and
        free_gpus

        Return
        ------
        None
        """
        self.status.update(
            pending=len(self.pending_tasks),
            running=len(self.running_tasks),
            completed=len(self.completed_tasks),
            failed=len(self.failed_tasks),
            free_cores=self.free_cores,
            free_gpus=self.free_gpus,
        )

        self.logger.info(
            "JOBS: Pending=%d Running=%d Completed=%d Failed=%d | RESOURCES: Free_cores=%d Free_gpus=%d",
            len(self.pending_tasks),
            len(self.running_tasks),
            len(self.completed_tasks),
            len(self.failed_tasks),
            self.free_cores,
            self.free_gpus,
        )

    def poolexecutor(
        self,
        task_arg_list: list[int | str],
        buffer_time: int | float = 0.5,
        adaptive: bool = True,
        dynopro: bool = False,
        submis_strat: TaskSubmissionStrategy | None = None,
        fut_proc_strat: FutureProcessingStrategy | None = None,
    ) -> None:
        """
        High-throughput executor implementation

        The poolexecutor uses the "Strategy Pattern" for a more concise and
        modular implementation of the poolexecutor. Depending on whether or not
        tasks make use of GPUs or only CPUs is adaptive or use the dynopro module
        the TaskSubmissionStrategy will be determined at run time and injected
        into the super loop.

        For implementing your own strategies you can look at the docs in the
        strategy directory of the matensemble package

        * GitHub: https://github.com/Q-CAD/MatEnsemble/blob/main/matensemble/matflux.py
        * GOF Design Patterns: https://en.wikipedia.org/wiki/Design_Patterns
        * Strategy Pattern: https://refactoring.guru/design-patterns/strategy

        Runs a 'super loop' until there are no more pending tasks and no
        running tasks.

        Each loop iteration:

        #. refreshes available resources
        #. prints a progress snapshot
        #. submits new jobs until resources are exhausted using a
        #. TaskSubmissionStrategy:
            * DynoproStrategy if dynopro=True
            * GPUAffineStrategy if gpus_per_task > 0
            * CPUAffineStrategy otherwise
        #. processes completed jobs using a FutureProcessingStrategy:
            * AdaptiveStrategy if adaptive=True
            * NonAdaptiveStrategy otherwise

        Parameters
        ----------
        task_arg_list: list[int | str]
            List of tasks to be scheduled and completed
        buffer_time: int | float
            The amount of time that will be used as the timeout= option for
            Future objects
        task_dir_list: list[str]
            Where completed tasks output files will be placed
        adaptive: bool
            Whether or not tasks are scheduled adaptively, default is True
        dynopro: bool
            Whether or not the dynopro module will be used for task submission,
            default is False

        Return
        ------
        None

        """

        # initialize submission strategy based on params at run-time

        if submis_strat is not None:
            submission_strategy = submis_strat
        elif dynopro:
            submission_strategy = DynoproStrategy(self)
        elif self.gpus_per_task > 0:
            submission_strategy = GPUAffineStrategy(self)
        else:
            submission_strategy = CPUAffineStrategy(self)

        # initialize future processing strategy at run-time
        if fut_proc_strat:
            future_processing_strategy = fut_proc_strat
        elif adaptive:
            future_processing_strategy = AdaptiveStrategy(
                self, gen_task_arg_list, gen_task_dir_list
            )
        else:
            future_processing_strategy = NonAdaptiveStrategy(self)

        # prepare resources
        self.flux_handle.rpc("resource.drain", {"targets": "0"}).get()
        with flux.job.FluxExecutor() as executor:
            # set executor in manager so that strategies can access it
            self.executor = executor

            self.logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
            start = time.perf_counter()

            done = len(self.pending_tasks) == 0 and len(self.running_tasks) == 0
            while not done:
                self.check_resources()
                self.log_progress()

                submission_strategy.submit_until_ooresources(
                    gen_task_arg_list, gen_task_dir_list, buffer_time
                )
                future_processing_strategy.process_futures(buffer_time)

                done = len(self.pending_tasks) == 0 and len(self.running_tasks) == 0

            end = time.perf_counter()
            self.logger.info("=== EXITING WORKFLOW ENVIRONMENT  ===")
            self.logger.info(f"Workflow took {(end - start):.4f} seconds to run.")
