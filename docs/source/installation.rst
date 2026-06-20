============
Installation
============

This guide covers the different methods of **installing** MatEnsemble: **what must be true** in your
environment and **copy-pastable patterns** for common HPC runtimes. Pair it with :doc:`tutorials`
for code samples and with :doc:`design` if you need a mental model of the runtime.

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

Tags follow the pattern ``<platform>-vX.Y.Z``:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Tag
     - Intended platform / notes
   * - ``pathfinder-vX.Y.Z``
     - Optimized for OLCF Pathfinder; pair with center module instructions.
   * - ``frontier-vX.Y.Z``
     - Optimized for OLCF Frontier; pair with center module instructions.
   * - ``perlmutter-vX.Y.Z``
     - Optimized for NERSC Perlmutter; pair with Shifter/Podman-HPC guidance from NERSC docs.
   * - ``linux-vX.Y.Z``
     - General install, not optimized for any specific system.

Containers are the recommended way to run MatEnsemble on HPC systems. If you are unfamiliar with HPC
container runtimes, see our overview of supported container engines for a brief introduction. Experienced
users can skip ahead to the installation and usage instructions for their target system.

Apptainer
---------

Apptainer (formerly Singularity) was developed at Lawrence Berkeley National Laboratory and is currently maintained by
The Linux Foundation. It is open source and is meant to be a container engine targeted
specifically for HPC systems. To get started you with MatEnsemble using Apptainer is simple.
You can build a Singularity Image Format (\*.sif) file that acts as a full portable container.
Apptainer is OCI compliant, so you can use our provided Docker images from the GitHub Container
Registry to build a \*.sif file.

.. code-block:: bash

    apptainer build <name>.sif docker://<source>/<image>:<tag>

The <name> is whatever you want the portable squashed image to be named and the <source>/<image>:<tag> will be
the version of MatEnsemble that you want to use. Here is an example of building a \*.sif file.

Apptainer images are immutable by design to ensure reproducibility, but you may want to add software
or packages into the image. Apptainer makes this very seamless, you can either convert your \*.sif file
into "sandbox" or you can build the sandbox from an OCI image published to a registry. The sandbox format
will allow you to install other packages or compile other software into the image. You can then convert
the changes you made into a transferrable \*.sif file that is immutable.

.. code-block:: bash


    apptainer build --sandbox <sandbox_name> docker://<source>/<image>:<tag>

    # or

    apptainer build --sandbox <sandbox_name> <path/to/sif>.sif


Once you have built the image you can install packages or compile software into it as if it were a Ubuntu
system. You just need to open it in an editable mode.

.. code-block:: bash

    apptainer shell --writable --cleanenv --fakeroot <path/to/sandbox>

After you have installed and changed all of the things that you want in the container you can then
squash it into an immutable format:

.. code-block:: bash

   apptainer build <name>.sif <path/to/sandbox>

For more information on how to build and manage apptainer images see `Introduction to Apptainer/Singularity <https://hsf-training.github.io/hsf-training-singularity-webpage/>`_.

Podman-HPC
----------

Podman-HPC is a wrapper around podman to allow it to be used on HPC systems. The NERSC Perlmutter system is
currently migrating from Shifter to Podman-HPC. If you are at all familiar with Docker then Podman-HPC
will feel very familiar to you as it uses all the same commands. You can pull any OCI image and
Podman-HPC will squash it automatically into a read only format and transfer it to your $SCRATCH storage

.. code-block:: bash

   podman-hpc pull <source>/<image>:<tag>

The main benefit of Podman-HPC is that as well as being a container runtime you can also use it as an
engine to build images. You can create a "recipe" as either a Dockerfile or more generally a Containerfile.
Then you can build that recipe.

.. code-block:: bash

   podman-hpc build -t <source>/<image>:<tag> .

When you first build the image it will not be put into a read-only format and it will not be migrated to
your $SCRATCH storage. So you need to either migrate to the $SCRATCH storage or push the image to a registry

.. code-block::

   # To migrate to $SCRATCH
   podman-hpc migrate <source>/<image>:<tag>

To push the image to a registry you first have to login to that respective registry and then you can enter
the "podman-hpc push" command.

If you want to add packages or compile other software into an image like you can with an apptainer
sandbox, you can. Its not as nice as with apptainer but its still possible. The most straightforward way
is to just edit our provided recipe to install whatever packages you want. You can find the recipe on
the `MatEnsemble GitHub Repository <https://github.com/FredDude2004/MatEnsemble/tree/main/example_workflows>`_.

You can also run an image in an interactive mode and install the packages and save the changes but the steps
are complicated and hard to get straight especially on HPC systems where you may lose connection.

Frontier (OLCF)
---------------

The OLCF Frontier super computer has Apptainer available to its users. So you can follow the
pattern for creating a container for Apptainer to create an environment to run MatEnsemble

.. code-block:: bash

    apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z

.. note::
   The Frontier image is quite large and squashing the image into the singularity image format
   may take a lot of time. It may be necessary to allocate yourself a compute node to build the
   container.

.. code-block:: bash

   salloc -A <project_id> -t 1:00:00 -N 1

   # connect to proxy server for internet access
   export all_proxy=socks://proxy.ccs.ornl.gov:3128/
   export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
   export http_proxy=http://proxy.ccs.ornl.gov:3128/
   export https_proxy=http://proxy.ccs.ornl.gov:3128/
   export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

   # build the container
   apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z

Tagged releases such as ``frontier-vX.Y.Z`` are the recommended images for users. Development images, when
published, may be more up to date but can be unstable. You can also build a sandbox in the same fashion.
You should make sure you are in your $SCRATCH space to make sure you have enough room for the sandbox
environment.

.. code-block:: bash

    # Example of building a sandbox for Frontier
    apptainer build --sandbox matensemble_sandbox docker://ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z

.. note::
   If building a sandbox is taking too long you can split it into multiple stages to speed things up.

.. code-block:: bash

   # first clean the cache to start fresh
   apptainer cache clean
   apptainer pull image.sif docker://ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z
   apptainer build ./matensemble_sand image.sif


After building your container you can run your workflows interactively in flux quite simply. You can optionally install
the MatEnsemble CLI tool to simplify the commands you need to run.

.. code-block:: bash

    # install the CLI tool to /usr/bin/
    curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash

After getting a SLURM allocation you can run your workflows:

.. code-block:: bash

   srun -N $SLURM_NNODES -n $SLURM_NNODES --external-launcher --mpi=pmi2 --pty apptainer exec matensemble.sif flux start

   # or with CLI tool

   matensemble set-image <path/to/sif_or_sandbox>
   matensemble shell                              # for an interactive session
   matensemble run <script.py>                    # to run a script non-interactively

In the interactive sessions you can verify that flux started properly with

.. code-block:: bash

   flux resource list

You should see all of the resources ready to use. You are ready run one of your matensemble scripts.

.. code-block:: bash

   python <script.py>

Perlmutter (NERSC)
------------------

To get MatEnsemble to work on Perlmutter you need to bind in the environment variables and devices that
allow the container to hook into the system's optimized network and MPI implementation manually. This can get ugly
quickly, especially when trying to work with Flux. So we provide a CLI tool to simplify this process for
the user. See our `batch script <https://github.com/FredDude2004/MatEnsemble/blob/main/example_workflows/perlmutter/lammps_mace/run_batch.slurm>`_
if you are curious.

To install the CLI tool you can run our install script:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash


After installation you can pull an image from our registry and allocate some nodes.

.. code-block:: bash

   # pull one of the perlmutter images for matensemble
   podman-hpc pull ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z

   # allocate yourself some nodes
   salloc -A <account_id> \
    -C gpu --qos=debug \
    -t <HH:MM:SS> \
    -N <number_of_nodes> \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --gpu-bind=closest

To use the tool you first configure the image that you want to run the workflow with and then run the
script that you want to execute.

.. code-block:: bash

   matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z
   matensemble run <script.py>

Pathfinder (OLCF)
-----------------

OLCF Pathfinder comes with Apptainer available to its users. You can build a container with the same pattern
as Frontier

.. code-block:: bash

    apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z

.. note::
   It may be necessary to allocate yourself a compute node to speed up the build.

.. code-block:: bash

   salloc -A <project_id> -t 1:00:00 -N 1

   # build the container
   apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z

.. code-block:: bash

    # Example of building a sandbox for Pathfinder
    apptainer build --sandbox matensemble_sandbox docker://ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z


You can run your workflows interactively in flux quite simply. You can either type the command yourself or
install our CLI tool to simplify the process.

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash


.. code-block:: bash

   srun -N $SLURM_NNODES -n $SLURM_NNODES --external-launcher --mpi=pmi2 --pty apptainer exec matensemble.sif flux start

   # or with CLI tool

   matensemble set-image <path/to/sif_or_sandbox>
   matensemble shell                              # for an interactive session
   matensemble run <script.py>                    # to run a script non-interacitvely

In the interactive sessions you can verify that flux started properly with

.. code-block:: bash

   flux resource list

You should see all of the resources ready to use and are ready run one of your matensemble scripts.

.. code-block:: bash

   python <script.py>

The curated Pathfinder smoke-test example lives under ``example_workflows/pathfinder/lammps_smoke``.

Conda
-----

We provide an environment.yaml file with all of the dependencies needed to run MatEnsemble (without GPU support).
If you have Anaconda or Miniconda installed and are on an x86_64 machine, then you can build an environment to
run MatEnsemble.

You can build a Conda environment with MatEnsemble and dependencies installed using the environment.yaml file.

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate matensemble

For more information see the `Anaconda Documentation <https://www.anaconda.com/docs/main>`__.

Dev Container
-------------

There is a .devcontainer folder in the repository so if you have Docker Desktop installed you can
simply clone out the repository and open it in VS Code with the devcontainer extension installed.
This will pull the general matensemble-base image and will then automatically install MatEnsemble
in an editable mode with all of its dependencies.

The image will have Flux, MPICH, and LAMMPS installed along with some extensions for Jupyter Notebooks.
In the CLI you can start a flux instance and a Jupyter server that has access to that.

.. code-block:: bash

   flux start --test-size=4 jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

After starting the flux allocation you can copy the link that is printed and register it as a
jupyter kernel in VS Code.

Where to read next
==================

* :doc:`tutorials` — minimal working programs.
* :doc:`examples` — curated repository examples by system.
* :doc:`reference` — exhaustive knobs (affinities, ``buffer_time``, dashboard port, failure reasons).
* :ref:`api-reference` — Sphinx autodoc for modules under ``src/matensemble``.
