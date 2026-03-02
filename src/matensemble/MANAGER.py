import flux

from matensemble.pipeline.PIPELINE import Job
from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy


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
        self.set

    def make_restart(self) -> None:
        pass

    def load_restart(self) -> None:
        pass

    def log_progress(self) -> None:
        pass

    def check_resources(self) -> None:
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
