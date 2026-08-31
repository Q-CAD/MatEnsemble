===
MCP
===

``mcp-matensemble`` is a small Model Context Protocol server for AI agents that
need to use MatEnsemble. It provides repository context, container
guidance, and dashboard helpers.

Install on an HPC login node
============================

The MCP server supports Frontier, Pathfinder, Perlmutter, and general Linux
HPC systems. To install the server you can use our script. The installer will prompt
you for the directory where the MatEnsemble workspace should be created.

.. code-block:: bash

    curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/refs/heads/main/install.sh | bash

The installer creates a MatEnsemble checkout and asks whether to write MCP
configuration files. If selected, it ensures that uv is installed and writes agent
configuration files for Claude Code, Codex, GitHub Copilot, Gemini, and VS Code into
a ``matensemble_campaigns`` workspace. By default it also pulls the latest image
for the selected system. Selecting Frontier, Pathfinder, or Perlmutter installs
and configures the matching site CLI automatically; general Linux installations
detect Apptainer, Docker, Podman, or Podman-HPC and do not install the HPC CLI.

Usage
=====

To make use of the MCP server you can use whichever agent you perfer. The installation
creates configurations for each of the frontier model's CLI tools

* `Claude Code <https://code.claude.com/docs/en/overview>`_
* `Codex <https://developers.openai.com/codex/cli>`_
* `Copilot <https://github.com/features/copilot/cli>`_
* `Gemini <https://geminicli.com/docs/>`_

Most of the CLI tools can be installed with a script with the exception of gemini needing
to be set up with anaconda on HPC systems.

Once you have one of these tools installed you can then start using the MatEnsemble MCP server

.. code-block:: bash

   # navigate to the campaigns directory
   cd /path/to/install/MatEnsemble/matensemble_campaigns

   # start the LLM with the CLI tool
   <claude, codex, copilot or gemini>

You can verify that the MCP server is running with

.. code-block:: bash

   /mcp

which will list the tools that are available to the agent. Or you can ask the agent
what is the most recent version of MatEnsemble to see if it has access to the tools
provided by the MCP server.

After it is configured you are free to use the agent to start building MatEnsemble
workflows.

Visual Studio Code
------------------

Along with the CLI configuartions there is a configuration for Visual Studio Code to launch the
MCP server. You can launch it by from the Command Pallete, "MCP: Server List" and you should see
the MatEnsemble MCP server available. You can then launch it and it will start the server as an
stdio server.

Tools
=====

The server exposes only:

* ``get_api_overview``
* ``get_containers_overview``
* ``get_examples_for_system``
* ``get_example_batch_scripts``
* ``get_containerfiles``
* ``get_container_build_command``
* ``get_matensemble_core``
* ``get_full_matensemble_code``
* ``get_matensemble_version``
* ``get_latest_container_tags``
* ``launch_dashboard``
* ``get_dashboard_access``
* ``stop_dashboard``

Dashboard
=========

There is an interactive dashboard which allows you to veiw the status of all past or running
workflows in the campaign directory. There is a prompt which is provided by the MCP server
to launch the dashboard. Simply ask the agent:

.. code-block:: text

   Can you use the MatEnsemble MCP server to launch the dashboard and give me the command to access it from localhost?

The agent will launch the dashboard on the login node and provide the command for you to
forward the port to localhost so that you can view your workflows.

Dashboard launches from the MCP server intentionally run through the MatEnsemble source
checkout with ``uv``:

.. code-block:: bash

   uv run --project /path/to/MatEnsemble matensemble-dashboard /path/to/matensemble_campaigns --host 127.0.0.1 --port 8000

The ``launch_dashboard`` response includes the exact command, working directory, project
root, and log path. If a launch exits immediately, check that the reported working
directory is the ``matensemble_campaigns`` directory and that the command starts with
``uv run --project <MatEnsemble checkout>``. The source checkout is used for uv project
resolution; the campaigns directory is used as the process working directory.
