===========
Quick Start
===========

MatEnsemble ships updated Docker images with all its dependencies packaged
and ready for use. To start using MatEnsemble you can pull one of our containers
down from the `registry <https://github.com/FredDude2004/MatEnsemble/pkgs/container/matensemble>`_

Tags
^^^^

*       matensemble:baseline-vX.Y.Z
*       matensemble:frontier-vX.Y.Z
*       matensemble:perlmutter-vX.Y.Z

Each tag is optimized for a specific HPC system. The baseline image is the most
general and portable container but it lacks the *hardware specific* GPU and MPI
capabilites that come with the other two images. 

Apptainer
---------

Many HPC systems come with a version of Apptainer (formerly Signularity) installed. 
Building MatEnsemble with apptainer is very simple. 

.. code-block:: bash

   apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:<tag>

.. note:: 
   We should put an example batch script here with commands

PodMan-HPC
----------

The NERSC Perlmutter system has been developing Podman-HPC as their container 
solution. Podman-HPC is greate because it can be used as a build engine as well
as a container runtime. The Perlmutter system also has Shifter available as a
runtime but they are planning on switching to Podman-HPC as their official container
solution so we recommend that you use that. 

.. note:: 
   For systems with PodMan-HPC instructions would go here. 


Frontier
--------

The OLCF Frontier system supports Apptainer as their container solution, which 
we provide a container that is compliant with Frontier's architecture. 

.. code-block:: bash 
    
   apptainer build matensemble.sif docker://ghcr.io/freddude2004/matenseble:frontier-vX.Y.Z

You can run the container with the following command 

.. code-block:: bash

   srun -N $SLURM_NNODES -n $SLURM_NNODES --pty --external-launcher --mpi=pmi2 --gpu-bind=closest apptainer exec matensemble.sif flux start


Perlmutter
----------

Put more detailed Perlmutter specific instructions here

MatEnsemble is available on PyPI and can be installed with pip

.. code-block:: bash

   pip install matensemble

MatEnsemble needs two pieces of the [flux-framework](https://flux-framework.readthedocs.io/en/latest/) installed to run. If you are on

MatEnsemble Without Containers
------------------------------

Containers are the easist way to get started with MatEnsemble, but if you would 
perfer not to use containers to run your MatEnsemble workflows then you can use
the official python package on `PyPI <link-to-MatEnsemble-package>`_. MatEnsemble
can be installed with pip 

.. code-block:: bash

   pip install matensemble[flux]

.. note::
   MatEnsemble needs two binaries of the flux-framework as well as LAMMPS available 
   and linked in order to run. So you would need to install them into an environment 
   as well as all of the python dependencies that MatEnsemble requires. 

.. warning:: 
   Installing MatEnsemble with this method has not yet been tested so be aware. 


