=========
Tutorials
=========

These examples assume you already have a **Flux session** and the ``matensemble`` package importable in
that environment in order to run (:doc:`installation`).

Here we will go over what chores are and how to create them, how to build a workflow, and some capabilities
that MatEnsemble gives you.

Minimal executable workflow
===========================

MatEnsemble is structured around :class:`~matensemble.chore.Chore`s which are units of work for MatEnsemble
to manage the state and execution of. There are two types of :class:`~matensemble.chore.Chore`s PYTHON and
EXECUTABLE. EXECUTABLE chores are the simpler of the two and can be created with the :class:`~matensemble.pipeline.Pipeline`

The Pipeline object will create a Directed Acyclic Graph (DAG) of these chore objects and once you submit
the graph to the manager it will sort the graph topologically and start the execution loop.

``Pipeline.exec`` records a :class:`~matensemble.chore.Chore`.
The command is either a string or an argv list.

.. code-block:: python
   :linenos:

   from matensemble.pipeline import Pipeline

   pipe = Pipeline()

   pipe.exec(command=["echo", "hello from MatEnsemble"])

   pipe.submit()

Nothing runs until :meth:`~matensemble.pipeline.Pipeline.submit`, which builds the DAG,
instantiates :class:`~matensemble.manager.FluxManager`, and enters the scheduling loop.

Parameters you will commonly set on :meth:`~matensemble.pipeline.Pipeline.exec`:

* ``num_tasks`` — Flux task count (for MPI programs this is usually your rank count).
* ``cores_per_task`` / ``gpus_per_task`` — resource hints for scheduling.
* ``mpi=True`` — toggles ``mpi=pmi2`` on the Flux jobspec; your program must initialize MPI accordingly.
* ``env`` / ``inherit_env`` — see :doc:`reference`.

PYTHON Chores and ``OutputReference``
=====================================

The other type of chores that you can create with MatEnsemble are PYTHON chores. These are still a unit
of work for MatEnsemble to manage the state and execution, but rather than a call to an external executable,
PYTHON chores are delayed function calls. You can define your own python functions and add those as chores
for the manager to handle.

The :class:`~matensemble.pipeline.Pipeline` has a decorator function that you can use to register a function.
When you instantiate a function it does NOT add any chores the the :class:`~matensemble.pipeline.Pipeline` yet.
Decorated functions are **not** executed immediately. When you call a PYTHON :class:`~matensemble.chore.Chore`
it returns a :class:`~matensemble.model.OutputReference` placeholder.

Defining chores
---------------

.. code-block:: python
   :linenos:

   from matensemble.pipeline import Pipeline
   from mpi4py import MPI

   # We first create a Pipeline and define an MPI-enabled chore that launches
   # 10 parallel MPI ranks using mpi4py.
   pipe = Pipeline()


   # Next we register a function to MatEnsemble
   @pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=True)
   def mpi_hello_world():
       size = MPI.COMM_WORLD.Get_size()
       rank = MPI.COMM_WORLD.Get_rank()
       name = MPI.Get_processor_name()

       print(f"Hello World! I am process {rank} of {size} on {name}.")


   # Then we create 10 Chore objects by calling the registered function
   for _ in range(10):
       mpi_hello_world()

   # Submit the workflow with the logger refreshing every second
   pipe.submit(log_delay=1)

Chaingin PYTHON Chores
======================

:class:`~matensemble.model.OutputReference`
objects can be treated as the results of a PYTHON chore and passed to other calls to PYTHON chores. When you pass
an :class:`~matensemble.model.OutputReference` to another chore call MatEnsemble will create an edge between those
two chores and when you submit the workflow MatEnsemble will see the downstream PYTHON chore as a dependent and will
ensure that these jobs are submitted in the correct order.

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
``chore2`` finishes. Internally, the worker deserializes ``../chore1/result.pickle`` before invoking ``chore2``.

.. note::

   Cycles are rejected during DAG validation. Fan-in (many tasks → one consumer) and fan-out are supported
   so long as the graph remains acyclic.

User Defined Strategies
=======================

MatEnsemble uses the strategy design pattern for the processing of chore completions. There are two
internal strategies that are shipped automatically. :class:`~matensemble.strategy.AdaptiveStrategy`
and :class:`~matensemble.strategy.NonAdaptiveStrategy`. The :class:`~matensemble.strategy.AdaptiveStrategy`
Users can also define their own strategies to be injected into the manager at runtime. MatEnsemble
provides another decorator to do this.

.. code-block:: python

    from matensemble.model import Resources
    from matensemble.pipeline import Pipeline
    from matensemble.chore import ChoreSpec

    pipe = Pipeline()

    screen_resources = dict(num_tasks=1, cores_per_task=1)
    validation_resources = dict(num_tasks=1, cores_per_task=4)

    @pipe.chore(name="screen_candidate", **screen_resources)
    def screen_candidate(candidate):
        """Cheap proxy for a simulation or surrogate-model evaluation."""
        temperature = candidate["temperature"]
        return {
            "candidate": candidate,
            "formation_energy": ((temperature - 1500) ** 2) / 1_000_000,
        }

    @pipe.chore(name="analyze_screen", **screen_resources)
    def analyze_screen(screen):
        """Decide whether this candidate deserves a more expensive validation."""
        energy = screen["formation_energy"]
        return {
            "candidate": screen["candidate"],
            "formation_energy": energy,
            "uncertainty": 0.12 if energy < 0.03 else 0.02,
        }

    @pipe.chore(name="validate_candidate", **validation_resources)
    def validate_candidate(candidate):
        """Placeholder for a larger MD, DFT, or phase-field validation run."""
        return {"candidate": candidate, "validation_status": "submitted"}

    @pipe.strategy(bolo_list=["analyze_screen"], **screen_resources)
    def request_validation(report):
        """Spawn validation only for uncertain, high-value candidates."""
        if report["uncertainty"] <= 0.05:
            return None

        return ChoreSpec(
            args=(report["candidate"],),
            kwargs={},
            resources=Resources(**validation_resources),
            qualname="validate_candidate",
        )

    for temperature in (1400, 1500, 1700):
        candidate = {"composition": "SiO2", "temperature": temperature}
        screen = screen_candidate(candidate)
        analyze_screen(screen)

    future = pipe.submit(log_delay=1)
    print(future.result())


The :meth:`~matensemble.pipeline.Pipeline.strategy` can be thought of as adding a callback to a
:class:`~matensemble.chore.Chore`. This function has access to the internal
:class:`~matensemble.manager.FluxManager` queue. If this strategy function returns a
:class:`~matensemble.chore.ChoreSpec` then it will be added to the matensemble queue at runtime.
This lets you create workflows that expand dynamically.

The ``bolo_list`` is the list of chore names that should trigger the strategy. If one of these chores completes,
MatEnsemble launches the strategy and passes the completed chore result as an argument.

User-defined strategies can observe completed chores and dynamically add more work by returning
:class:`~matensemble.chore.ChoreSpec` objects.

Nested arguments
================

Dependency scanning walks nested containers and non-class dataclasses. You may pass structured payloads
mixing plain data and :class:`~matensemble.model.OutputReference` instances; the worker recursively replaces
references with concrete Python objects.

Third-party imports inside chores
=================================

Because workers import the defining module in full, **top-level imports** run automatically. You do not need
to bury ``import numpy`` inside the chore body unless you want lazy loading for side-effect control.

If you need extra wheels:

* **Containers:** extend the provided image (Apptainer ``%post`` snippet with ``uv pip install …``,
  :doc:`installation`).
* **Virtualenv on NFS:** install once into the environment shared by all nodes.

Further reading
===============

* :doc:`reference` — complete ``submit()`` parameter table and artifact schemas.
* :ref:`api-reference` — authoritative signatures mirrored from the source docstrings.
