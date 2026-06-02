=========
Tutorials
=========

These examples assume you already have a **Flux session** and the ``matensemble`` package importable in
that environment (:doc:`installation`).

Example repositories
====================

Reference implementations live under ``example_workflows/`` in the `MatEnsemble GitHub repository
<https://github.com/FredDude2004/MatEnsemble/tree/main/example_workflows>`__. Paths worth opening first:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Example
     - What it demonstrates
   * - ``mpi_example.py``
     - Demonstrates how to construct a :class:`~matensemble.pipeline.Pipeline` and create a PYTHON Chore
   * - ``dependency_example.py``
     - Demonstrates how to create a dependency chain and submit it.
   * - ``strategy_injection_example.py``
     - An example of how to create a User Defined Strategy and attatch it to another chore.

Minimal executable (“exec”) workflow
====================================

``Pipeline.exec`` records a :class:`~matensemble.chore.Chore
The command is either a string (split with :mod:`shlex`) or an argv list.

.. code-block:: python
   :linenos:
   from matensemble.pipeline import Pipeline

   pipe = Pipeline()

   pipe.exec(command=["echo", "hello from MatEnsemble"])

   pipe.submit()

Nothing runs until :meth:`~matensemble.pipeline.Pipeline.submit`, which builds the DAG (trivial here),
instantiates :class:`~matensemble.manager.FluxManager`, and enters the scheduling loop.

Parameters you will commonly set on :meth:`~matensemble.pipeline.Pipeline.exec`:

* ``num_tasks`` — Flux task count (for MPI programs this is usually your rank count).
* ``cores_per_task`` / ``gpus_per_task`` — resource hints for scheduling.
* ``mpi=True`` — toggles ``mpi=pmi2`` on the Flux jobspec; your program must initialize MPI accordingly.
* ``env`` / ``inherit_env`` — see :doc:`reference`.

**Dependencies:** Executable chores created through :meth:`~matensemble.pipeline.Pipeline.exec` **do not**
inspect :class:`~matensemble.model.OutputReference` objects; they are always treated as roots unless you wrap
external commands inside a Python chore.

Python chores and ``OutputReference`` dependencies
=================================================

Decorated functions are **not** executed immediately. Each call appends a Python :class:`~matensemble.chore.Chore`
and returns a :class:`~matensemble.model.OutputReference` placeholder.

Defining chores (importable module — **not** ``__main__``)
--------------------------------------------------------

.. code-block:: python
   :linenos:

   from matensemble.pipeline import Pipeline
   from mpi4py import MPI

   # We first create a Pipeline and define an MPI-enabled chore that launches
   # 10 parallel MPI ranks using mpi4py.
   pipe = Pipeline()


   @pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=True)
   def mpi_hello_world():
       size = MPI.COMM_WORLD.Get_size()
       rank = MPI.COMM_WORLD.Get_rank()
       name = MPI.Get_processor_name()

       print(f"Hello World! I am process {rank} of {size} on {name}.")


   # Then we add the chore to the workflow 10 seperate times
   for _ in range(10):
       mpi_hello_world()

   # in 10 separate MPI jobs being executed through the matensemble workflow
   # runtime and scheduler backend.


   pipe.submit(log_delay=1)

Chained dependencies (any acyclic DAG)
======================================

.. code-block:: python

   # functions.py
   from matensemble.pipeline import Pipeline

   pipe = Pipeline()

   @pipe.chore()
   def chore1():
       return 1

   @pipe.chore()
   def chore2(x):
       return x + 1

   @pipe.chore()
   def chore3(x):
       return x * 2

   a = chore1()
   b = chore2(a)
   c = chore3(b)

   pipe.submit()

:class:`~matensemble.manager.FluxManager` only schedules ``chore2`` after ``chore1`` finishes, and ``chore3`` after
``chore2`` finishes. Internally, the worker unpickles ``../chore1/result.pkl`` before invoking ``chore2``.

.. note::

   Cycles are rejected during DAG validation. Fan-in (many tasks → one consumer) and fan-out are supported
   so long as the graph remains acyclic.

User Defined Strategies
=======================

Coming soon

Nested arguments
================

Dependency scanning walks nested containers and non-class dataclasses. You may pass structured payloads
mixing plain data and :class:`~matensemble.model.OutputReference` instances; the worker recursively replaces
references with concrete Python objects.

Third-party imports inside chores
===============================

Because workers import the defining module in full, **top-level imports** run automatically. You do not need
to bury ``import numpy`` inside the chore body unless you want lazy loading for side-effect control.

If you need extra wheels:

* **Containers:** extend the provided image (Apptainer ``%post`` snippet with ``uv pip install …``,
  :doc:`installation`).
* **Virtualenv on NFS:** install once into the environment shared by all nodes.

Operational tips
================

* Pass ``dashboard=True`` and tunnel port ``8000`` if you want the browser UI (:doc:`architecture`).
* Inspect ``matensemble_workflow.log`` for human-readable progress; parse ``status.json`` for machine consumption.
* On failure, always read the chore’s ``stderr``—MatEnsemble annotates wrapper errors there.

Further reading
===============

* :doc:`reference` — complete ``submit()`` parameter table and artifact schemas.
* :ref:`api-reference` — authoritative signatures mirrored from the source docstrings.
