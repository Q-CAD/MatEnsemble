.. _design:

Design and execution model
=================================

This page describes **what runs where** when you call :meth:`matensemble.pipeline.Pipeline.submit`,
how MatEnsemble talks to Flux, and how results move between tasks. It complements the
step-by-step tutorials in :doc:`tutorials`.

Runtime prerequisites
---------------------

MatEnsemble assumes you are already inside a **Flux allocation** (or another environment
where ``flux.Flux()`` can attach to a running broker). Typical patterns on HPC systems:

* Submit an interactive or batch job that runs ``flux start`` (or your site’s equivalent)
  and then launches your Python driver inside that session.
* The workflow driver process imports MatEnsemble, builds a :class:`~matensemble.pipeline.Pipeline`,
  and calls :meth:`~matensemble.pipeline.Pipeline.submit`.

The Python package on PyPI does **not** replace the need for **flux-core** / sched binaries
provided by your center. The ``flux`` optional dependency installs the **Python bindings**
(``flux-python``) that talk to those libraries. See :doc:`installation`.

Objects you interact with
-------------------------

:class:`~matensemble.pipeline.Pipeline`
    Builder for a directed acyclic graph (DAG) of :class:`~matensemble.chore.Chore` instances.
    Calling ``@pipe.chore``-decorated functions records delayed work; :meth:`~matensemble.pipeline.Pipeline.exec`
    adds shell/executable work.

:class:`~matensemble.chore.Chore`
    Immutable specification for a single Flux submission: command vector, resource request,
    working directory, and (for Python chores) pickels callable by value into the **registry**

:class:`~matensemble.manager.FluxManager`
    Created when you call :meth:`~matensemble.pipeline.Pipeline.submit`. It owns queues of ready,
    blocked, and running chore IDs, tracks free cores/GPUs from Flux, and drives the main scheduling loop.

:class:`~matensemble.fluxlet.Fluxlet`
    Thin wrapper that turns a :class:`~matensemble.chore.Chore` into a Flux ``JobspecV1`` and submits it
    through a :class:`flux.job.FluxExecutor`.

``matensemble.runtime_worker``
    A normal Python module launched **as the Flux job command** for PYTHON-type chores. It unpickles
    the chore, imports your function, substitutes dependency results, runs the function, and writes
    ``result.pickle``.


Workflow directory layout
-------------------------

When you construct :class:`~matensemble.pipeline.Pipeline`, it picks a timestamped root under your
chosen base directory (by default, the current working directory):

.. code-block:: text

   <base>/
   └── matensemble_workflow-YYYYMMDD_HHMMSS/
       ├── status.json              # atomically updated workflow summary
       ├── status_history.jsonl     # append-only dashboard history
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

The string ``<base>`` is :meth:`pathlib.Path.cwd` unless you pass ``basedir=`` to :class:`~matensemble.pipeline.Pipeline`.
The workflow folder name uses a compact timestamp.


DAG construction and ordering
------------------------------

Edges in the DAG are derived solely from :class:`~matensemble.model.OutputReference` placeholders
embedded in a Python chore’s positional or keyword arguments (including nested tuples, lists, dicts,
and non-class dataclass instances). :meth:`matensemble.pipeline.Pipeline.exec` does **not** currently
accept dependency references; treat executable chores as root tasks unless you wrap shell work inside
a Python chore.

Before submit, MatEnsemble:

#. Builds a :class:`networkx.DiGraph` with an edge ``upstream → downstream`` for each dependency.
#. Verifies that every referenced chore ID exists.
#. Rejects cycles (topological sort must succeed).

The :class:`~matensemble.manager.FluxManager` receives chores in topological order, but **submission**
order is additionally constrained by live resource availability (cores and GPUs).


Resource accounting
-------------------

Each chore declares :class:`~matensemble.model.Resources`:

* ``num_tasks`` — Flux task count for the chore.
* ``cores_per_task`` — CPU cores per task.
* ``gpus_per_task`` — GPUs per task (may be zero).

The manager estimates **needed** cores and GPUs as ``num_tasks * cores_per_task`` and
``num_tasks * gpus_per_task``, and compares against:

* The **total** allocation (all chores must fit in the worst case—oversized chores are marked invalid).
* The **currently free** counts reported by Flux after rank 0 is drained for the broker.

GPU affinity shell options are only applied when ``gpus_per_task > 0`` and GPU affinity is enabled
on submit.

Main scheduling loop (“super loop”)
-----------------------------------

Roughly each iteration:

#. Refresh Flux free resource counts.
#. Write ``status.json`` and a log line with pending / running / completed / failed counts.
#. Drain the **ready** queue and submit every chore that fits; defer the rest to the back of the queue.
#. Wait up to ``buffer_time`` seconds for at least one Flux future to complete (strategy-dependent).
#. For each finished future: interpret exit code / exceptions, update dependents, and (in adaptive
   mode) try to submit more work immediately.

The two built-in strategies are :class:`~matensemble.strategy.AdaptiveStrategy` and
:class:`~matensemble.strategy.NonAdaptiveStrategy`; see :doc:`reference` for behavioral differences.

Failure propagation
-------------------

If a chore fails submission, raises in the Flux future wrapper, or returns a non-zero process exit code,
MatEnsemble records a failure and **cascades** to all transitive dependents so the workflow cannot deadlock.
Downstream chores receive failure reason ``dependency_failed`` with an ``upstream`` chore ID in the internal
failure list. Check per-chore ``stderr`` for the detailed MatEnsemble annotations written by
:class:`~matensemble.strategy`.

Dashboard (optional)
--------------------

Pass ``dashboard=True`` to :meth:`~matensemble.pipeline.Pipeline.submit`. A Starlette + uvicorn thread
serves static assets and ``GET /api/status`` on **port 8000**. On a cluster you typically **SSH tunnel**
from your laptop to the compute node running the driver—for example:

.. code-block:: bash

   ssh -L 8000:<nodelist>:8000 <user>@<login.host>

Use the exact hostname of the node where your workflow process runs; the snippet above is only illustrative.
