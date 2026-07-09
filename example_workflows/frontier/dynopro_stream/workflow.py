"""
Frontier dynopro smoke workflow.

This example registers two ordinary MatEnsemble chores and uses
Pipeline.dynopro() to run them as rank-colored subprocesses:

- gpu_stream_producer runs on the GPU-colored ranks.
- cpu_stream_consumer runs on the CPU-colored ranks and receives producer
  payloads over MPI.

Frontier nodes expose 8 GPU devices as MI250X GCDs and 64 CPU cores. The Flux
broker uses one allocated node, so with ``#SBATCH -N 2`` this workflow requests
one dynopro worker node by default. Override the defaults with
MATENSEMBLE_DYNOPRO_NNODES, MATENSEMBLE_GPUS_PER_NODE, or
MATENSEMBLE_CORES_PER_NODE if your allocation layout is different.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from matensemble.pipeline import Pipeline

GPUS_PER_NODE = int(os.environ.get("MATENSEMBLE_GPUS_PER_NODE", "8"))
CORES_PER_NODE = int(os.environ.get("MATENSEMBLE_CORES_PER_NODE", "64"))
STREAM_TAG = 4100

pipe = Pipeline()


def _worker_nodes() -> int:
    explicit = os.environ.get("MATENSEMBLE_DYNOPRO_NNODES")
    if explicit is not None:
        return int(explicit)

    slurm_nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    if slurm_nodes is None:
        return 1

    return max(int(slurm_nodes) - 1, 1)


def _local_rank(comm) -> tuple[int, int, int]:
    rank = comm.Get_rank()
    local_rank = rank % CORES_PER_NODE
    node_index = rank // CORES_PER_NODE
    node_base = node_index * CORES_PER_NODE
    return rank, local_rank, node_base


def _visible_gpu() -> str:
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if value:
            return f"{key}={value}"
    return "no GPU visibility variable set"


@pipe.chore(name="gpu_stream_producer")
def gpu_stream_producer(comm=None, split=None, chore_dir: Path | None = None) -> dict:
    rank, local_rank, node_base = _local_rank(comm)
    target_rank = node_base + GPUS_PER_NODE + local_rank

    payload = {
        "producer_rank": rank,
        "producer_local_rank": local_rank,
        "target_rank": target_rank,
        "visible_gpu": _visible_gpu(),
        "timestamp": time.time(),
        "message": f"hello from GPU-colored rank {rank}",
    }

    if target_rank < comm.Get_size():
        comm.send(payload, dest=target_rank, tag=STREAM_TAG + local_rank)
        print(f"GPU rank {rank} streamed payload to CPU rank {target_rank}: {payload}")
    else:
        print(f"GPU rank {rank} had no paired CPU rank; payload was {payload}")

    return payload


@pipe.chore(name="cpu_stream_consumer")
def cpu_stream_consumer(comm=None, split=None, chore_dir: Path | None = None) -> dict:
    rank, local_rank, node_base = _local_rank(comm)
    cpu_slot = local_rank - GPUS_PER_NODE

    if 0 <= cpu_slot < GPUS_PER_NODE:
        source_rank = node_base + cpu_slot
        payload = comm.recv(source=source_rank, tag=STREAM_TAG + cpu_slot)
        record = {
            "consumer_rank": rank,
            "consumer_local_rank": local_rank,
            "source_rank": source_rank,
            "received": payload,
        }
        print(f"CPU rank {rank} received stream from GPU rank {source_rank}: {payload}")
    else:
        record = {
            "consumer_rank": rank,
            "consumer_local_rank": local_rank,
            "source_rank": None,
            "received": None,
            "idle": True,
        }

    if chore_dir is not None:
        stream_log = Path(chore_dir) / f"cpu-stream-rank-{rank}.txt"
        with stream_log.open("w") as f:
            f.write(repr(record))
            f.write("\n")

    return record


pipe.dynopro(
    gpu_subprocess="gpu_stream_producer",
    cpu_subprocess="cpu_stream_consumer",
    nnodes=_worker_nodes(),
    gpus_per_node=GPUS_PER_NODE,
    cores_per_node=CORES_PER_NODE,
    name="frontier-dynopro-stream",
)

future = pipe.submit(log_delay=5)
print(future.result())
