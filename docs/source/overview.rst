========
Overview
========

MatEnsemble is a framework to build, orchestrate, and asynchronously manage scalable workflows, especially targeted for compute-intensive AI-driven materials modeling simulations (e.g., atomistic modeling, Phase-Field, etc.) as efficiently as possible.
Apart from standard automated high-throughput computations, the core of MatEnsemble is designed to support "user-defined" acquisition strategies to dynamically steer workflows based on intermediate results, which is a common pattern in active learning and other autonomous workflows at scale.
To enable extremely scalable parametric sweeps and bypass standard scheduler bottlenecks, typically encountered in leadership computing platforms, MatEnsemble uses a single large allocation and an internal scheduler to manage arbitrarily large workloads. The library is built on top of the Flux resource manager, which provides efficient job scheduling and resource management capabilities, making it well-suited for high-throughput computing scenarios.


MatEnsemble benefits from the native Python executor interface of ``Flux`` [ahn2020flux]_
and the concurrent asynchronous programming model of core Python through
``concurrent.futures.Future`` objects [quinlan2009futures]_. Continuous throughput is maintained
by dynamically spawning and monitoring tasks. Furthermore, to enable real-time streaming of
post-processed data from large-scale atomistic trajectories, MatEnsemble uses an *in-memory*
data analysis protocol that exploits the heterogeneous GPU+CPU architecture of exascale systems
(e.g., Frontier) via a round-robin MPI communicator-splitting approach (cf. [bagchi2025matensemble]_).
As explained in the following sections, such an online adaptive framework efficiently couples
available computing resource chunks for ensemble evaluations guided by adaptive sampling methods.


Scalable and adaptive scheduling
=========================================

In the context of high-throughput materials modeling, fully leveraging exascale resource capabilities with SLURM or similar schedulers is challenging due to:

* Many short ``sbatch`` / ``srun`` invocations increase scheduler load and log volume.
* Queue latency dominates when tasks are tiny relative to scheduler quanta.
* Some centers cap how many job steps you may launch inside a single allocation.

A common mitigation is **one large allocation** plus an **internal scheduler** that launches many child
processes or MPI ranks inside that allocation. The remaining problem is **utilization**: if you launch work
in static waves, fast tasks finish early and cores sit idle while slow tasks run. MatEnsemble addresses that
with its **adaptive** task orchestration capability, new/pending tasks are launched as soon as resources free up, keeping the allocation saturated until all work is done.


What MatEnsemble does in one sentence
=====================================

Inside a **Flux** session, MatEnsemble continuously: (1) tracks available computing resources, (2) submits ready DAG
nodes in the queue, (3) processes completed flux jobs, (4) unblocks dependents, and (5) repeats until no ready,
running, or blocked work remains and/or spawns new DAGs depending on user-defined strategies.

See :doc:`architecture` for the exact loop, artifacts, and environment assumptions.

Core concepts
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

References
==========

.. [ahn2020flux] Ahn, D. H., Bass, N., Chu, A., Garlick, J., Grondona, M., Herbein, S.,
   Ingolfsson, H. I., Koning, J., Patki, T., Scogland, T. R. W., Springmeyer, B.,
   and Taufer, M. (2020). "Flux: Overcoming scheduling challenges for exascale workflows."
   *Future Generation Computer Systems*, 110, 202-213. https://doi.org/10.1016/j.future.2020.04.006

.. [quinlan2009futures] Quinlan, B. (2009). "PEP 3148 -- futures - execute computations
   asynchronously." Python Enhancement Proposals. https://peps.python.org/pep-3148/

.. [bagchi2025matensemble] Bagchi, S., Biswas, A., Balachandran, P. V., Ghosh, A.,
   and Ganesh, P. (2025). "Towards 'on-demand' van der Waals epitaxy with an adaptive
   resource-driven online ensemble sampling simulation framework." arXiv:2504.05539.
   https://doi.org/10.48550/arXiv.2504.05539

Next steps
==========

* :doc:`quickstart` — containers, PyPI install, site-specific shell snippets.
* :doc:`tutorials` — minimal Python and executable examples, dependency graphs, packaging tips.
* :doc:`architecture` — Flux interactions, ``PYTHONPATH``, failure propagation, dashboard tunneling.
* :doc:`reference` — exhaustive parameter and artifact listing.
* :ref:`api-reference` — docstring-generated module documentation.
