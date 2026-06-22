========
Overview
========

MatEnsemble is a **workflow manager** for running many :class:`~matensemble.chore.Chore` instances on a
supercomputer as efficiently as possible.

MatEnsemble is a framework to build, orchestrate, and asynchronously manage scalable adaptive-learning
workflows, especially targeted for compute-intensive AI-driven materials modeling simulations
(e.g., atomistic modeling, Phase-Field, etc.) as efficiently as possible. Apart from standard automated high-throughput
computations, the core of MatEnsemble is designed to support "user-defined" acquisition strategies to
dynamically steer workflows based on intermediate results, which is a common pattern in active learning
and other autonomous workflows at scale. To enable scalable parametric sweeps and bypass standard scheduler
bottlenecks, typically encountered in leadership computing platforms, MatEnsemble uses a single large
allocation and an internal scheduler to manage arbitrarily large workloads. The library is built on top
of the Flux resource manager, and implements an efficient and user-defined-strategy based task orchestration
protocol, making it well-suited for high-throughput autonomous computing scenarios.

MatEnsemble [bagchi2025matensemble]_ benefits from the native Python executor interface of **Flux**
[ahn2020flux]_, and the concurrent asynchronous programming model of core Python through
``concurrent.futures`` objects [quinlan2009futures]_.

You build a directed acyclic graph (DAG) of :class:`~matensemble.chore.Chore`
objects in Python; MatEnsemble submits work to **Flux**, tracks completions, records logs, and keeps hardware busy
while tasks finish at different rates.

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

**Inside your Flux session**, MatEnsemble repeatedly: (1) reads free CPU/GPU counts, (2) submits ready chores,
(3) processes completed chores, (4) unblocks dependents, and (5) repeats until no ready, running, or blocked work remains.

See :doc:`design` for the exact loop, artifacts, and environment assumptions.

Core concepts
=============

:class:`~matensemble.pipeline.Pipeline`
    User-facing builder. Python functions decorated with :meth:`~matensemble.pipeline.Pipeline.chore` turn into delayed function calls; :meth:`~matensemble.pipeline.Pipeline.exec`
    adds argv-style work to the :obj:`matensemble.pipeline.Pipeline`.

:class:`~matensemble.model.OutputReference`
    Placeholder returned from a delayed function call. Passing it into another chore encodes a **dependency edge**
    and ensures upstream results are unpickled before the downstream function runs.

:class:`~matensemble.chore.Chore`
    Single Flux submission record—command, resources, working directory, and for PYTHON chores, the qualified name of the function you want to call.

:class:`~matensemble.manager.FluxManager`
    Runtime coordinator created when you call :meth:`~matensemble.pipeline.Pipeline.submit`.

:class:`~matensemble.strategy.FutureProcessingStrategy`
    Pluggable completion handler. Built-ins: :class:`~matensemble.strategy.AdaptiveStrategy` (fill idle
    resources as tasks finish) and :class:`~matensemble.strategy.NonAdaptiveStrategy` (wave-style drain).

Logging and on-disk layout
==========================

Every run creates a **timestamped workflow directory** under your chosen base path (by default the current working directory):

.. code-block:: text

   <base>/
   └── matensemble_workflow-YYYYMMDD_HHMMSS/
       ├── status.json              # atomically updated workflow summary
       ├── status_history.jsonl     # append-only dashboard history
       ├── matensemble_workflow.log # detailed text log from the ``matensemble`` logger
       └── out/
           ├── registry/            # serialized chore callables
           │   ├── Callable name
           │   ├── Callable name
           │   └── ...
           └── <chore_id>/
               ├── stdout
               ├── stderr
               ├── chore.pickle     # serialized chore object
               ├── metadata.json    # metadata of the chore in JSON for debugging
               └── result.pickle    # serialized python chore return value

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

.. image:: ../../media/chain_v_adaptive_scheduling.png
   :alt: Diagram contrasting static batching with adaptive back-filling of tasks

User Defined Strategies
=======================

MatEnsemble uses the *strategy pattern* when processing :class:`flux.job.FluxExecutorFuture` completions:

Users can define their own strategies to be injected into the processing loop. Say for instance you wanted to
spawn more :class:`~matensemble.chore.Chore` objects dynamically (while the workflow is already running) based on the
results of a certain :class:`~matensemble.chore.Chore`. You can define a function that takes the results of a
:class:`~matensemble.chore.Chore` and performs your processing on it and returns a :class:`~matensemble.chore.ChoreSpec`
which will be dynamically added to the submissions queue. Here is an example of adding a strategy to a chore

.. code-block:: python

    import random

    from matensemble.chore import ChoreSpec
    from matensemble.model import Resources
    from matensemble.pipeline import Pipeline

    pipe = Pipeline()

    @pipe.chore()
    def generate_num():
        return random.randint(1, 1000)

    @pipe.chore()
    def fizz(n):
        print(f"{n} is divisible by 3")
        print("fizz")

    @pipe.chore()
    def buzz(n):
        print(f"{n} is divisible by 5")
        print("buzz")

    @pipe.chore()
    def fizzbuzz(n):
        print(f"{n} is divisible by 3 and 5")
        print("fizzbuzz")

    @pipe.strategy(bolo_list=["generate_num"])
    def proc_strat(results_of_finished_chore):
        if results_of_finished_chore % 15 == 0:
            return ChoreSpec(
                        args=(results_of_finished_chore,),
                        kwargs=None,
                        resources=Resources(),
                        qualname="fizzbuzz"
                    )
        elif results_of_finished_chore % 5 == 0:
            return ChoreSpec(
                        args=(results_of_finished_chore,),
                        kwargs=None,
                        resources=Resources(),
                        qualname="buzz"
                    )
        elif results_of_finished_chore % 3 == 0:
            return ChoreSpec(
                        args=(results_of_finished_chore,),
                        kwargs=None,
                        resources=Resources(),
                        qualname="fizz"
                    )
        else:
            print(f"{results_of_finished_chore} is not divisible by 3 or 5")

    for _ in range(10):
        generate_num()

    pipe.submit()

The :obj:`bolo_list` is telling the manager which chores it should Be-On-the-LookOut for. So whenever the
manager sees a *generate_num* chore instance complete then it will spawn the user defined strategy as a new
chore. This strategy can optionally return a :obj:`matensemble.chore.ChoreSpec` which will spawn a new chore
with the specified args kwargs and resources (cores, gpus, mpi, etc.).

Roadmap and stability
=====================

.. note::

   The project is under active development; pre-1.0 APIs may still move. Track ``CHANGELOG`` / release notes
   in the repository for breaking changes. The internal **dynopro** package (streaming dynamics and heavy
   analysis) ships in-tree but is not yet part of the curated Sphinx API toctree.

**Checkpointing:** ``write_restart_freq`` exists on :meth:`~matensemble.pipeline.Pipeline.submit`, but
checkpoint serialization is **not yet implemented**. Long production runs should pass ``None`` until
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

* :doc:`installation` — containers, PyPI install, site-specific shell snippets.
* :doc:`tutorials` — minimal Python and executable examples, dependency graphs, packaging tips.
* :doc:`design` — Flux interactions, ``PYTHONPATH``, failure propagation, dashboard tunneling.
* :doc:`reference` — exhaustive parameter and artifact listing.
* :ref:`api-reference` — docstring-generated module documentation.
