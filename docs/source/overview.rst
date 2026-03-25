========
Overview
========

MatEnsemble is a **workflow manager** for running many similar :class:`~matensemble.chore.Chore` instances on a
supercomputer as efficiently as possible. You build a directed acyclic graph (DAG) in Python; MatEnsemble
submits work to **Flux**, tracks completions, records logs, and keeps hardware busy while tasks finish
at different rates.

The library targets **high-throughput** and **ensemble** scenarios: thousands of small simulations,
parameter sweeps, or analysis pipelines where classic “one Slurm job per task” workflows would overwhelm the
scheduler or spend too much time queued.

High-throughput computing and schedulers
=========================================

High Throughput Computing (HTC) maximizes completed work over long windows by running **many independent**
tasks, often modest in size. Sites frequently call this pattern **task farming**.

Farming through raw Slurm has costs:

* Many short ``sbatch`` / ``srun`` invocations increase scheduler load and log volume.
* Queue latency dominates when tasks are tiny relative to scheduler quanta.
* Some centers cap how many job steps you may launch inside a single allocation.

A common mitigation is **one large allocation** plus an **internal scheduler** that launches many child
processes or MPI ranks inside that allocation. The remaining problem is **utilization**: if you launch work
in static waves, fast tasks finish early and cores sit idle while slow tasks run. MatEnsemble addresses that
with **adaptive** submission tied to live Flux resource reporting.

What MatEnsemble does in one sentence
=====================================

**Inside your Flux session**, MatEnsemble repeatedly: (1) reads free CPU/GPU counts, (2) submits ready DAG
nodes that fit, (3) processes completed Flux jobs, (4) unblocks dependents, and (5) repeats until no ready,
running, or blocked work remains.

See :doc:`architecture` for the exact loop, artifacts, and environment assumptions.

Core concepts (with pointers)
==============================

:class:`~matensemble.pipeline.Pipeline`
    User-facing builder. Decorated Python functions turn into delayed chores; :meth:`~matensemble.pipeline.Pipeline.exec`
    adds argv-style work.

:class:`~matensemble.model.OutputReference`
    Placeholder returned from a delayed Python call. Passing it into another chore encodes a **dependency edge**
    and ensures upstream results are unpickled before the downstream function runs.

:class:`~matensemble.chore.Chore`
    Single Flux submission record—command, resources, working directory, and (for Python chores) pointers back
    to your source module.

:class:`~matensemble.manager.FluxManager`
    Runtime coordinator created when you call :meth:`~matensemble.pipeline.Pipeline.submit`.

:class:`~matensemble.strategy.FutureProcessingStrategy`
    Pluggable completion handler. Built-ins: :class:`~matensemble.strategy.AdaptiveStrategy` (fill idle
    resources as tasks finish) and :class:`~matensemble.strategy.NonAdaptiveStrategy` (wave-style drain).

Logging and on-disk layout
==========================

Every run creates a **timestamped workflow directory** under your chosen base path (by default the current
working directory):

.. code-block:: text

   <base>/matensemble_workflow-YYYYMMDD_HHMMSS/
   ├── status.json                 # compact counters for dashboards / monitoring
   ├── matensemble_workflow.log    # verbose rolling log from the ``matensemble`` logger
   └── out/
       └── <chore_id>/
           ├── stdout
           ├── stderr
           ├── chore.pkl / chore.json  # serialized chore specification & debug view
           └── result.pkl / result.json   # Python return values only

The driver prints a short hint to stderr with absolute paths to ``status.json``, the log file, and the ``out``
tree when logging initializes.

Adaptive vs non-adaptive scheduling
===================================

In **adaptive** mode (the default), completing a chore can **immediately** trigger more submissions in the same
super-loop iteration via :meth:`~matensemble.manager.FluxManager._submit_until_ooresources`, keeping the
allocation saturated when backlog exists.

In **non-adaptive** mode, the manager only submits during the initial “fill until out of resources” phases;
completion handling updates the DAG but **does not** proactively pull additional ready chores until the next
outer-loop scheduling opportunity—use this when you want tighter control or simpler resource snapshots.

.. image:: ../../images/Cap_1_adaptive_task_management.png
   :alt: Diagram contrasting static batching with adaptive back-filling of tasks

Strategies
==========

MatEnsemble uses the *strategy pattern* when processing :class:`flux.job.FluxExecutorFuture` completions:

.. code-block:: python

   class FutureProcessingStrategy(ABC):
       @abstractmethod
       def process_futures(self, buffer_time) -> None:
           ...

Concrete implementations live in :mod:`matensemble.strategy`. Supply your own subclass to
:meth:`~matensemble.pipeline.Pipeline.submit` if you need custom batching, metrics hooks, or integration with
external planners—ensure your strategy maintains the manager’s queue invariants or you may deadlock.

Roadmap and stability
=====================

.. note::

   The project is under active development; pre-1.0 APIs may still move. Track ``CHANGELOG`` / release notes
   in the repository for breaking changes. The internal **dynopro** package (streaming dynamics and heavy
   analysis) ships in-tree but is not yet part of the curated Sphinx API toctree.

**Checkpointing:** ``write_restart_freq`` exists on :meth:`~matensemble.pipeline.Pipeline.submit`, but
checkpoint serialization is **not implemented** yet. Long production runs should pass ``None`` until
restart files are supported (:doc:`reference`).

Next steps
==========

* :doc:`quickstart` — containers, PyPI install, site-specific shell snippets.
* :doc:`tutorials` — minimal Python and executable examples, dependency graphs, packaging tips.
* :doc:`architecture` — Flux interactions, ``PYTHONPATH``, failure propagation, dashboard tunneling.
* :doc:`reference` — exhaustive parameter and artifact listing.
* :ref:`api-reference` — docstring-generated module documentation.
