============
Installation
============

This guide covers the different methods of **installing** MatEnsemble. **what must be true** in your
environment and **copy-pastable patterns** for common HPC runtimes. Pair it with :doc:`tutorials`
for code samples and with :doc:`architecture` if you need a mental model of the runtime.

Versions and compatibility
==========================

* **Python:** ``>=3.12`` (see ``requires-python`` in the project metadata).
* **Flux:** You need a working Flux allocation or single-user Flux instance **before** importing MatEnsemble
  for real runs. The PyPI extra ``flux`` installs the Python bindings; it does not install flux-core for you.
* **Operating system:** Linux is assumed for HPC-style Flux workflows. macOS or Windows installs may work for editing
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
     - Most portable; use when you do not need a site-tuned image.
   * - ``frontier-vX.Y.Z``
     - Optimized for OLCF Frontier; pair with center module instructions.
   * - ``perlmutter-vX.Y.Z``
     - Optimized for NERSC Perlmutter; pair with Shifter/Podman-HPC guidance from NERSC docs.

Apptainer
---------

Apptainer (formerly Singularity) was developed at Lawerence Berkeley National Laboratory and is currently maintained by
The Linux Foundation. It is open source and is meant to be a container engine targeted
specifically for HPC systems. Using Apptainer is simple. To get started you can build with
MatEnsemble using Apptainer is simple. You can build a Singularity Image Format (\*.sif) file
that acts as a full portable container. Apptainer is OCI compliant, so you can use our provided
Docker images from the GitHub Container Registry to build a \*.sif file.

.. code-block:: bash

    apptainer build <name>.sif docker://<image>:<tag>

The <name> is whatever you want the portable squashed image to be named and the image tag will be
the version of MatEnsemble that you want to use. Here is an example of building a \*.sif file for
Frontier

.. code-block:: bash

    apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:frontier-dev

The frontier-dev tag will be the most up to date version of MatEnsemble which is updated with each
push to main, but may be unstable.

Apptainer images are immutable by design to ensure reproducibility, but you may want to add software
or packages into the image. Apptainer makes this very seamless, instead of building an immutable
image you can build a "sandbox" which will allow you to install other packages or compile other
software into the image. You can then convert the changes you made into a transferrable \*.sif file
that is immutable

.. code-block:: bash

    apptainer build --sandbox <sandbox_name> docker://<image>:<tag>

    # Example of building a sandbox for Frontier
    apptainer build --sanbox matensemble_sandbox docker://ghcr.io/freddude2004/matensemble:frontier-dev

Once you have built the image you can install packages or compile software into it as if it were a Ubuntu
system. You just need to open it in an editable mode.

.. code-block:: bash

    apptainer shell --writable --cleanenv --fakeroot matensemble_sandbox

For more information on how to build and manage apptainer images see `Introduction to Apptainer/Singularity <https://hsf-training.github.io/hsf-training-singularity-webpage/>`_.

Podman-HPC
----------

Podman-HPC a wrapper around podman to allow it to be used on HPC systems. The NERSC Perlmutter system is
currently migrating from Shifter to Podman-HPC. If you are at all familiar with Docker then Podman-HPC
will feel very familiar to you as it uses all the same commands.

Frontier (OLCF) example skeleton
--------------------------------

Build the SIF (see above) with the ``frontier-vX.Y.Z`` tag. A typical pattern is to request nodes, start
Flux, then execute the container under Slurm with PMI-aware MPI options. **Always verify against the current**
`Frontier user guide <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html>_`.

.. code-block:: bash
   :caption: example batch script for Frontier

    # First load the frontier helper modules for apptainer compatibility with GPUs and system MPICH
    module load olcf-container-tools
    module load apptainer-enable-mpi
    module load apptainer-enable-gpu

    srun -N $SLURM_NNODES -n $SLURM_NNODES --external-launcher --mpi=pmi2 apptainer exec matensemble.sif flux start <your-workflow-command>


Replace ``<your-workflow-command>`` with something like ``python <script>.py`` that constructs a
:class:`~matensemble.pipeline.Pipeline` and calls :meth:`~matensemble.pipeline.Pipeline.submit`.
If you want something more interactive then you can allocate a node with salloc and run an interacitive teletype

.. code-block:: bash
   :caption: example batch script for Frontier

    # Request an interactive allocation on Frontier
    salloc \
      --account=<PROJECT_ID> \
      --partition=debug \
      --nodes=<NUM_NODES> \
      --time=<WALLTIME>

    # Shorter syntax
    salloc -A <PROJECT_ID> -p debug -N <NUM_NODES> -t <WALLTIME>

Once you have the allocation you can start an interactive flux instance:

.. code-block:: bash

    srun -N $SLURM_NNODES -n $SLURM_NNODES --pty --external-launcher --mpi=pmi2 apptainer exec matensemble.sif flux start

    # Verify that the allocation sees all the resources
    flux resource list

Perlmutter (NERSC)
------------------

To get MatEnsemble to work on Perlmutter you have to do some pretty hacky stuff. So for the best results you should
follow our `examples for Perlmutter <https://github.com/FredDude2004/MatEnsemble/tree/main/example_workflows/perlmutter>`_.

Conda
-----

.. warning::
   This has not been tested. Eventually when we have an official MatEnsemble package on PyPI you will be able to install via Conda

Dev Container
-------------

There is a .devcontainer folder in the repository so if you have Docker Desktop install you can
simply clone out the repository and open it in VS Code with the devcontainer extension installed.
This will pull the matensemble-base image made for baseline and will then automatically install
MatEnsemble in an editable mode with all of its dependencies.

The image will have Flux, MPICH, and LAMMPS installed along with some extensions for Jupyter Notebooks.
In the CLI you can start a flux instance and a Jupyter server that has access to that.

.. code-block:: bash

   flux start --test-size=4 jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

After starting the flux allocation you can copy the link that is printed and register it as a
jupyter kernel in VS Code.

Where to read next
==================

* :doc:`tutorials` — minimal working programs.
* :doc:`reference` — exhaustive knobs (affinities, ``buffer_time``, dashboard port, failure reasons).
* :ref:`api-reference` — Sphinx autodoc for modules under ``src/matensemble``.
