=====================
MatEnsemble MCP Agent
=====================

``mcp-matensemble`` is a separate Model Context Protocol server that helps AI
agents create MatEnsemble workflow campaigns, plan site environments, and submit
validated Slurm batch scripts.

Install on an HPC login node
============================

To install the mcp server for a system you first have to login to the cluster. The
server is managed with uv so that also needs to be installed, then you can run our
script to install the server.

.. code-block:: bash

    # install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # clone out MatEnsemble
    git clone https://github.com/freddude2004/MatEnsemble.git
    cd MatEnsemble

    uv run --package mcp-matensemble matensemble-agent-install --system <system>

When you run the script it will create a directory $SCRATCH/matensemble_campaigns/
which will have a configuration for vscode to be able to launch the MCP server. The
script will also install the matensemble CLI tool for your system.

Tool safety model
=================

The MCP server writes only under the configured workspace. Execution tools are
dry-run by default.

Allowed setup commands are limited to known environment tools such as
``apptainer``, ``podman-hpc``, ``docker``, ``podman``, ``conda``, and the
MatEnsemble site CLI.

Slurm submission is available through a guarded ``sbatch`` tool. The script must
be inside the workspace, end with ``.slurm``, contain ``#SBATCH`` directives, and
appear to launch a MatEnsemble workflow.

Starter prompts
===============

.. code-block:: text

   I want to run a LAMMPS GPU smoke test with MatEnsemble on Frontier. Use the MatEnsemble MCP server to inspect examples, plan setup, and create workflow and launch scripts. Do not execute setup commands or submit jobs yet.

.. code-block:: text

   I want to run a LAMMPS/MACE campaign with MatEnsemble on Perlmutter. Use the Perlmutter examples as context, plan the Podman-HPC image setup, and create a smoke campaign with a batch script. Do not submit yet.

.. code-block:: text

   Review this generated campaign, validate the Slurm script, and dry-run the sbatch command.
