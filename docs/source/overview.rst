========
Overview
========

MatEnsemble is a **workflow manager** for running many :class:`~matensemble.chore.Chore` instances on a
supercomputer as efficiently as possible. You build a directed acyclic graph (DAG) of :class:`~matensemble.chore.Chore`s in Python; MatEnsemble
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

What MatEnsemble does
=====================

**Inside your Flux session**, MatEnsemble repeatedly: (1) reads free CPU/GPU counts, (2) submits ready DAG
nodes that fit, (3) processes completed Flux jobs, (4) unblocks dependents, and (5) repeats until no ready,
running, or blocked work remains.

See :doc:`design` for the exact loop, artifacts, and environment assumptions.

Core concepts
=============

:class:`~matensemble.pipeline.Pipeline`
    User-facing builder. Decorated Python functions turn into delayed function calls; :meth:`~matensemble.pipeline.Pipeline.exec`
    adds argv-style work.

:class:`~matensemble.model.OutputReference`
    Placeholder returned from a delayed funciton call. Passing it into another chore encodes a **dependency edge**
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

   <base>/
   └── matensemble_workflow-YYYYMMDD_HHMMSS/
       ├── status.json              # atomically updated for the dashboard / monitoring
       ├── matensemble_workflow.log # detailed text log from the ``matensemble`` logger
       └── out/
           ├── registry/            # pickled chore callables
           │   ├── Callable name
           │   ├── Callable name
           │   └── ...
           └── <chore_id>/
               ├── stdout
               ├── stderr
               ├── chore.pickle     # Pickled chore object
               ├── metadata.json    # Metadata of the chore in JSON for debugging
               └── result.pickle    # Python chore return value (pickle)

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

User Defined Strategies
=======================

MatEnsemble uses the *strategy pattern* when processing :class:`flux.job.FluxExecutorFuture` completions:

.. code-block:: python

   class FutureProcessingStrategy(ABC):
       @abstractmethod
       def process_futures(self, buffer_time) -> None:
           ...

Users can define their own strategies to be injected into the processing loop. Say for instance you wanted to
spawn more :class:`~matensemble.chore.Chore`s dynamically (while the workflow is already running) based on the
results of a ceratin :class:`~matensemble.chore.Chore`. You can define a funciton that takes the results of a
:class:`~matensemble.chore.Chore` and performs your processing on it and returns a :class:`~matensemble.chore.ChoreSpec`
which will be dynamically added to the submissions queue. Which can effectively add cycles to the workflow if
needed.

Roadmap and stability
=====================

.. note::

   The project is under active development; pre-1.0 APIs may still move. Track ``CHANGELOG`` / release notes
   in the repository for breaking changes. The internal **dynopro** package (streaming dynamics and heavy
   analysis) ships in-tree but is not yet part of the curated Sphinx API toctree.

Next steps
==========

* :doc:`installation` — containers, PyPI install, site-specific shell snippets.
* :doc:`tutorials` — minimal Python and executable examples, dependency graphs, packaging tips.
* :doc:`design` — Flux interactions, ``PYTHONPATH``, failure propagation, dashboard tunneling.
* :doc:`reference` — exhaustive parameter and artifact listing.
* :ref:`api-reference` — docstring-generated module documentation.
