import flux

from matensemble.pipeline.PIPELINE import Job
from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy
from matensemble.FLUXLET import Fluxlet


class Manager:
    def __init__(
        self, 
        *, 
        write_restart_freq: int | None = 100, 
        nnodes: int | None = None, 
        gpus_per_node: int | None = None, 
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
    ) -> None:
        self.flux_handle = flux.Flux
        self.fluxet = Fluxlet(self.flux_handle)

        self.write_restart_freq = write_restart_freq
        # self.set

    def make_restart(self) -> None:
        """
        Pickle the current state of the manager and dump it to a file
        """
        pass

    def load_restart(self) -> None:
        """
        Load the pickled restart file and pick up where it left off. 
        """
        pass

    def log_progress(self) -> None:
        """
        Update the status file and append a progress line in the log file
        """
        pass

    def check_resources(self) -> None:
        """
        See what resources are available with flux
        """
        pass

    def poolexecutor(
        self,
        job_list: list[Job],
        *,
        buffer_time: int | None = None,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
