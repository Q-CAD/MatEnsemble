=========
Tutorials
=========

These examples assume you already have a **Flux session** and the ``matensemble`` package importable in
that environment (:doc:`quickstart`).

Example repositories
====================

Reference implementations live under ``example_workflows/`` in the `MatEnsemble GitHub repository
<https://github.com/FredDude2004/MatEnsemble/tree/main/example_workflows>`__. Paths worth opening first:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Path
     - What it demonstrates
   * - ``hello_world/exec_workflow.py``
     - Ten executable chores calling the same Python helper script with different arguments.
   * - ``hello_world/chore_workflow.py`` / ``run_chore_workflow.py``
     - Split-module pattern for decorated Python chores.
   * - ``dependency_example/``
     - Tiny DAG showing chained return values.

Minimal executable (“exec”) workflow
====================================

``Pipeline.exec`` records a :class:`~matensemble.chore.Job` with chore type :attr:`~matensemble.model.ChoreType.EXECUTABLE`.
The command is either a string (split with :mod:`shlex`) or an argv list.

.. code-block:: python

   from pathlib import Path

   from matensemble.pipeline import Pipeline

   pipe = Pipeline()

   pipe.exec(command=["/bin/echo", "hello from MatEnsemble"])

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

Batch of executable chores (from the hello_world example)
==========================================================

.. code-block:: python

   import sys
   from pathlib import Path

   from matensemble.pipeline import Pipeline

   pipe = Pipeline()
   script = Path(__file__).with_name("mpi_helloworld.py")

   for i in range(1, 11):
       pipe.exec(command=[sys.executable, str(script), str(i)], num_tasks=50)

   pipe.submit()

Here ``num_tasks=50`` launches 50 Flux tasks; combine with ``mpi=True`` when your script expects PMI.

Python chores and ``OutputReference`` dependencies
=================================================

Decorated functions are **not** executed immediately. Each call appends a Python :class:`~matensemble.chore.Job`
and returns a :class:`~matensemble.model.OutputReference` placeholder.

Defining chores (importable module — **not** ``__main__``)
--------------------------------------------------------

.. code-block:: python

   # workflow.py  (module name must be importable, e.g. package.workflow)
   from mpi4py import MPI
   from matensemble.pipeline import Pipeline

   pipe = Pipeline()

   @pipe.chore()
   def run_mpi_hello(task_id: int):
       size = MPI.COMM_WORLD.Get_size()
       rank = MPI.COMM_WORLD.Get_rank()
       name = MPI.Get_processor_name()

       out = f"task_{task_id}_rank_{rank}.txt"
       with open(out, "w", encoding="utf-8") as f:
           f.write(f"Hello from rank {rank}/{size} on {name}, task={task_id}\n")

       return rank

Driver script (this **may** be ``__main__``)
----------------------------------------------

.. code-block:: python

   # run_workflow.py
   from workflow import pipe, run_mpi_hello

   def main():
       for i in range(1, 11):
           run_mpi_hello(i)

       pipe.submit()

   if __name__ == "__main__":
       main()

Why this split matters
----------------------

Python chores store ``func_module`` and ``func_qualname``. If you define chores in the same file you later run
with ``python that_file.py``, Python sets ``__name__ == "__main__"`` and ``func_module`` becomes
``"__main__"``. The remote worker then imports ``__main__`` in a different process context, which **fails**
or picks up the wrong definitions.

.. warning::

   Define decorated chores in a **regular module** (for example ``workflow.py`` or ``pkg/tasks.py``) and import
   them from a tiny runner script. This requirement may be relaxed in a future release, but it is mandatory
   today.

Recommended file layout
-----------------------

.. code-block:: text

   project/
   ├── my_workflow.py    # Pipeline + @pipe.chore definitions
   └── run.py            # imports my_workflow, builds graph, calls pipe.submit()

You can add ``__init__.py`` if you place code inside a package; ensure the working directory / ``PYTHONPATH``
story from :doc:`architecture` still resolves.

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

.. code-block:: python

   # run_workflow.py
   from functions import pipe, chore1, chore2, chore3

   a = chore1()
   b = chore2(a)
   c = chore3(b)

   pipe.submit()

:class:`~matensemble.manager.FluxManager` only schedules ``chore2`` after ``chore1`` finishes, and ``chore3`` after
``chore2`` finishes. Internally, the worker unpickles ``../chore1/result.pkl`` before invoking ``chore2``.

.. note::

   Cycles are rejected during DAG validation. Fan-in (many tasks → one consumer) and fan-out are supported
   so long as the graph remains acyclic.

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
  :doc:`quickstart`).
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
