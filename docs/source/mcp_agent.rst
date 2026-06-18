=====================
MatEnsemble MCP Agent
=====================

``mcp-matensemble`` is a separate Model Context Protocol server that helps AI
agents create MatEnsemble workflow campaigns, plan site environments, and submit
validated Slurm batch scripts.

Install on an HPC login node
============================

The recommended installation is the site-aware bootstrap script:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/scripts/install-matensemble-agent.sh | bash -s -- --system frontier

Use ``--system perlmutter`` or ``--system pathfinder`` for other systems.

The installer:

* installs ``mcp-matensemble`` with ``uv tool install``
* installs the MatEnsemble site CLI as ``matensemble``
* creates ``$SCRATCH/matensemble_campaigns`` when ``$SCRATCH`` exists
* writes ``.vscode/mcp.json`` in the campaign workspace
* writes a small workspace README and MCP config file

Open the generated campaign workspace with VS Code Remote SSH. The MCP server
runs on the remote system and writes files into that workspace.

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
