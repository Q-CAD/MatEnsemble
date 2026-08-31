=========
Tutorials
=========

Here we walk you through how to start a flux session and begin running workflows interactively
with MatEnsemble. We assume that you have already installed MatEnsemble on whatever system you
are planning to run on (if not you can refer to :doc:`installation`).

.. contents:: Contents
   :depth 1

Starting Flux
=============

Flux is a resource manager much like slurm, but is entirely in user space. Flux gives users the
ability to have fine grained control over their allocation, allowing them to run jobs on individual
CPU cores, GPUs and entire nodes hierarchically.

.. contents:: Systems
   :local:
   :depth: 1

Locally
-------

If you have flux installed locally then you can start flux with:

.. code-block:: bash

   flux start

This will launch an interactive flux session and give you a shell into the environment.
You can see what resources flux is seeing with:

.. code-block:: bash

   flux resource list

Which will print a message showing the number of nodes, cores and gpus that flux sees.
If you have downloaded the general linux container image for MatEnsemble then you will
first need to spin up a container from the image. Depending on what container engine
you are running the command will differ slightly (refer to :doc:`installation` for more
details). Here is how to start a container with docker.

.. code-block:: bash

   docker run --rm -it ghcr.io/q-cad/matensemble:linux-vX.Y.Z /bin/bash

This will give you a shell into a single use docker container where you can then run:

.. code-block:: bash

   flux start

To put begin a flux session.

Frontier
--------

On the OLCF Frontier system you will need a conainer setup in order to run flux. The system
provided module is outdated and does not have the most recent features of flux that MatEnsemble
makes use of. Follow our :doc:`installation` instructions for Frontier to get a container
ready on Frontier.

Once you have a SIF file ready you can start flux on either a login node or under a SLURM allocation
on a compute node

Login Node
~~~~~~~~~~

Go to the directory where your SIF file is located and run the command:

.. code-block:: bash

   apptainer exec <name>.sif flux start

This will give you a shell into the flux session. You can verify with:

.. code-block:: bash

   flux resource list

Which will print out the resources that flux is seeing.

Compute Node
~~~~~~~~~~~~

In order to start flux on a compute node you can run the following command:

.. code-block:: bash

   srun \
    -N $SLURM_NNODES \
    -n $SLURM_NNODES \
    --external-launcher \
    --gpu-bind=closest \
    --mpi=pmi2 \
    apptainer exec <name>.sif flux start

This will give you an interactive shell into the flux session which can see the entire SLURM
allocation. You can verify by running:

.. code-block:: bash

   flux resource list

Pathfinder
----------

Pathfinder is virtually identical to Frontier. You can start a flux session on a login node
for some quick testing or on a compute node.

Login
~~~~~

Go to the directory where your SIF file is located and run the command:

.. code-block:: bash

   apptainer exec <name>.sif flux start

This will give you a shell into the flux session. You can verify with:

.. code-block:: bash

   flux resource list

Which will print out the resources that flux is seeing.

Compute
~~~~~~~

In order to start flux on a compute node you can run the following command:

.. code-block:: bash

   srun \
    -N $SLURM_NNODES \
    -n $SLURM_NNODES \
    --external-launcher \
    --mpi=pmi2 \
    apptainer exec <name>.sif flux start

This will give you an interactive shell to the flux session which can see all of your
resources. You can verify with.

.. code-block:: bash

   flux resource list

Perlmutter
----------

Currently it is not recommended to start an interactive flux session on Perlmutter, it is very unstable
and note very user friendly. Instead the recommended approach is to use the MatEnsemble CLI tool. If you
are really curious then you can take a look at the `cli script <https://github.com/FredDude2004/MatEnsemble/blob/main/src/cli/matensemble-perlmutter>_`
for Perlmutter.

Running MatEnsemble
===================

After you are in a flux session you can now run a MatEnsemble script. Here we will go over the
general shape of MatEnsemble workflows and common patterns.

Pipeline
--------

The main user facing APIs of MatEnsemble are accesed through the Pipeline object. The pipline object
is used to define work for the manager to schedule, add strategies and launch dynamic and adaptive
workflows.

.. code-block:: python

   from matensemble.pipeline import Pipeline

   pipe = Pipeline(
            basedir=None,
            registry=None,
            reserve_broker_node=None,
            controller_cores=None,
          )

Accepting the defaults is the recommended use of MatEnsemble but some of the arguments are worth
noting depending on your use case. If you are on a single node machine (i.e. your local computer)
then the reserve_broker_node should be set to False. MatEnsemble defaults to reserving an entire
node to run the manager. If you are only on one node this wouldn't leave any resources for the
actual jobs to run. Instead you can have the manager run on a specified number of CPU cores.

Chores
------

Executable
~~~~~~~~~~

Python
~~~~~~

Strategy
--------

Running on Perlmutter
=====================

Here is a walkthrough on how to run MatEnsemble on Perlmutter using the MatEnsemble CLI tool.
In order to have MatEnsemble run properly on Perlmutter there has to be a lot of extra directories
and libraries bound into the container at runtime which is impractical for users to enter by hand.
For guides on how to create workflow files and how to run MatEnsemble read the above documentation.

To install the MatEnsemble CLI tool you can run our installation script.

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/main/src/cli/install.sh | bash

This will place an executable script at `/usr/bin/matensemble`.

Next you need to pull the MatEnsemble container image for Perlmutter:

.. code-block:: bash

   podman-hpc pull ghcr.io/q-cad/matensemble:perlmutter-vX.Y.Z

In order to run MatEnsemble on Perlmutter you need to be on a GPU node. The image has drivers that
get bound into the image that will fail if no NVIDIA GPU is detected. You can get an allocation with

.. code-block:: bash

   salloc \
    -A <project-name> \
    -C gpu \
    --qos=debug \
    -t HH:MM:SS \
    -N <num-nodes> \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --gpu-bind=closest

Once you have an allocation you will then need to set the container image with:

.. code-block:: bash

   matensemble set-image ghcr.io/q-cad/matensemble:perlmutter-vX.Y.Z

With everything in place you can now navigate to wherever you have a MatEnsemble workflow script and
run it with:

.. code-block:: bash

   matensemble run <script-name>.py
