=====================
MatEnsemble MCP Agent
=====================

``mcp-matensemble`` is a small Model Context Protocol server for AI agents that
need to use MatEnsemble. It provides repository context, deterministic container
guidance, and dashboard lifecycle helpers.

It intentionally does not generate campaigns, submit Slurm jobs, cancel jobs,
write workflow files, or execute container setup plans. The agent should use the
context returned by this server, then write and run ordinary files and commands
itself.

Install on an HPC login node
============================

From the repository root, run:

.. code-block:: bash

   ./install.sh

The installer prompts for Frontier, Perlmutter, or Pathfinder, chooses an install
root, clones MatEnsemble into ``<root>/MatEnsemble/.matensemble``, creates
``<root>/MatEnsemble/matensemble_campaigns``, optionally installs the site CLI,
and writes MCP config files for Codex, Claude Code, VS Code, and Copilot.

The generated MCP command has this shape:

.. code-block:: bash

   uv run --directory <root>/MatEnsemble/.matensemble --package mcp-matensemble mcp-matensemble --system <system>

Tools
=====

The server exposes only:

* ``get_api_overview``
* ``get_containers_overview``
* ``get_examples_for_system``
* ``get_containerfiles``
* ``get_container_build_command``
* ``get_matensemble_core``
* ``get_full_matensemble_code``
* ``get_matensemble_version``
* ``get_latest_container_tags``
* ``launch_dashboard``
* ``get_dashboard_access``
* ``stop_dashboard``

Container tags
==============

The server does not query GHCR. It derives the expected current image tags from
the local MatEnsemble version:

.. code-block:: text

   ghcr.io/freddude2004/matensemble:<system>-vX.Y.Z

Dashboard
=========

``launch_dashboard`` starts ``matensemble dashboard`` on the login node and
writes PID/log files into the campaign root. ``get_dashboard_access`` returns the
SSH tunnel command to run on a local machine, and ``stop_dashboard`` terminates a
dashboard started by the MCP server.
