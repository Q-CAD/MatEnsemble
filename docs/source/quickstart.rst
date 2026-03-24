===========
Quick start
===========

This guide covers **how to obtain** MatEnsemble (containers vs PyPI), **what must be true** in your
environment (Flux session, Python version, optional MPI), and **copy-pastable patterns** for common HPC
runtimes. Pair it with :doc:`tutorials` for code samples and with :doc:`architecture` if you need a mental
model of the runtime.

Versions and compatibility
==========================

* **Python:** ``>=3.12`` (see ``requires-python`` in the project metadata).
* **Flux:** You need a working Flux allocation or single-user Flux instance **before** importing MatEnsemble
  for real runs. The PyPI extra ``flux`` installs the Python bindings; it does not install flux-core for you.
* **Operating system:** Linux is assumed for HPC-style Flux workflows. macOS installs may work for editing
  workflows but are not the primary target for execution.

Container images (recommended on clusters)
===========================================

Official images are published to GitHub Container Registry:

`ghcr.io/freddude2004/matensemble <https://github.com/FredDude2004/MatEnsemble/pkgs/container/matensemble>`__

Tags follow the pattern below (replace ``X.Y.Z`` with the release you want—which should match, or be
compatible with, the ``version`` field in ``pyproject.toml`` in the same commit):

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Tag
     - Intended platform / notes
   * - ``baseline-vX.Y.Z``
     - Most portable GPU/MPI stack; use when you do not need a site-tuned image.
   * - ``frontier-vX.Y.Z``
     - Optimized for OLCF Frontier; pair with center module instructions.
   * - ``perlmutter-vX.Y.Z``
     - Optimized for NERSC Perlmutter; pair with Shifter/Podman-HPC guidance from NERSC docs.

Pulling with Docker (laptop or build host)
------------------------------------------

.. code-block:: bash

   docker pull ghcr.io/freddude2004/matensemble:baseline-vX.Y.Z

Apptainer / Singularity ``.sif`` from the registry
---------------------------------------------------

Most DOE and academic systems provide Apptainer. Convert the OCI image locally or on a login node:

.. code-block:: bash

   apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:baseline-vX.Y.Z

Run **inside an allocation that already has Flux running**, after ``cd`` into the directory that contains
your workflow sources so ``PYTHONPATH`` resolution (see :doc:`architecture`) matches your layout.

Podman-HPC (e.g. NERSC Perlmutter transition)
---------------------------------------------

Podman-HPC can act as both build engine and runtime. Follow your center’s current “containers on Perlmutter”
documentation; specifics change as Shifter → Podman-HPC rollouts progress. General pattern:

#. Load the site’s Podman-HPC module or wrapper.
#. Pull or build the MatEnsemble image exactly as the center documents for **MPI + GPU** visibility.
#. Launch **under Flux** using the center’s example that wires ``flux run`` / ``srun`` compatibility layers.

Because invocation flags differ by facility, this documentation intentionally stays descriptive—copy the
**GPU binding**, **network**, and **process launch** flags from a **recent** NERSC or vendor cookbook and swap
the image URL for ``ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z``.

Frontier (OLCF) example skeleton
--------------------------------

Build the SIF (see above) with the ``frontier-vX.Y.Z`` tag. A typical pattern is to request nodes, start
Flux, then execute the container under Slurm with PMI-aware MPI options. **Always verify against the current**
`Frontier user guide <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html>`__—the snippet below is
only structural:

::

   # Inside a batch script AFTER resources are allocated:
   srun -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --pty --external-launcher --mpi=pmi2 --gpu-bind=closest \
     apptainer exec matensemble.sif flux start <your-workflow-command>

Replace ``<your-workflow-command>`` with something like ``python driver.py`` that constructs a
:class:`~matensemble.pipeline.Pipeline` and calls :meth:`~matensemble.pipeline.Pipeline.submit`.

PyPI install (editable or virtualenv)
=====================================

`matensemble on PyPI <https://pypi.org/project/matensemble/>`__

.. code-block:: bash

   pip install "matensemble[flux]"

The ``flux`` extra pins ``flux-python`` to the version declared in project metadata. You still need **system**
``flux-core`` / ``flux-sched`` (or a modules tree providing them) that matches what your center expects.

.. warning::

   Fully reproducible **non-container** installs require you to line up: Python 3.12+, Flux C libraries,
   MPI if you use ``mpi=True`` jobs, optional GPU runtimes, and any simulation binaries you call from
   executable jobs (for example LAMMPS). Treat the PyPI package as the **Python layer** only.

If you do **not** need Flux bindings locally (for example, you are only editing workflows):

.. code-block:: bash

   pip install matensemble

Local docs build (developers)
-----------------------------

From a clone of the repository, with dev dependencies (``uv`` example):

.. code-block:: bash

   uv sync --group dev
   uv run sphinx-build -b html docs/source docs/build

Open ``docs/build/index.html`` in a browser.

Verification checklist before your first run
=============================================

#. ``flux getattr version`` or your site’s health check succeeds on the node.
#. ``python -c "import flux, matensemble"`` works in the **same environment** your job will use.
#. You launch the driver from the directory you intend to be the **workflow parent** (see :doc:`architecture`).
#. For Python jobs, job functions live in an **importable module** that is not executed as ``__main__`` (see :doc:`tutorials`).
#. For long runs, set ``write_restart_freq=None`` until restart support ships (:doc:`reference`).

Where to read next
==================

* :doc:`tutorials` — minimal working programs.
* :doc:`reference` — exhaustive knobs (affinities, ``buffer_time``, dashboard port, failure reasons).
* :ref:`api-reference` — Sphinx autodoc for modules under ``src/matensemble``.
