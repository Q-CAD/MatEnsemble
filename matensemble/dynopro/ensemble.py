from matensemble.manager import SuperFluxManager
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
        write_restart_freq (int): Frequency to write restart files.
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
        write_restart_freq=1000,
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

    def run(self):
        """
        Run the ensemble dynamics simulations.

        This method initializes the SuperFluxManager and executes the simulations in parallel.
        """
        # Initialize SuperFluxManager
        sfm = SuperFluxManager(
            gen_task_list=self.sim_list,
            gen_task_cmd=self.sim_command,
            ml_task_cmd=None,
            tasks_per_job=self.tasks_per_job,
            cores_per_task=self.cores_per_task,
            gpus_per_task=self.gpus_per_task,
            write_restart_freq=self.write_restart_freq,
            nnodes=self.nnodes,
            gpus_per_node=self.gpus_per_node,
        )

        # Execute the simulations
        sfm.poolexecutor(
            task_arg_list=self.sim_args_list,
            buffer_time=self.buffer_time,
            task_dir_list=self.sim_dir_list,
            adaptive=self.adaptive,
            dynopro=True,
        )

