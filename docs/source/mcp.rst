===
MCP
===

``mcp-matensemble`` is a small Model Context Protocol server for AI agents that
need to use MatEnsemble. It provides repository context, container
guidance, and dashboard helpers.

Install on an HPC login node
============================

Currently the MCP server is only targeted at the Frontier, Pathfinder and Perlmutter
HPC systems. To install the server you can use our script. Before running the install
script make sure the $SCRATCH environment variable is set or navigate to a location
that would like your workflows to live.

.. code-block:: bash

    curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/refs/heads/main/install.sh | bash

The installer will create a directory named MatEnsemble which is where most of the
MCP configuration will live. It will install the MatEnsemble CLI tool to ~/.local/bin
and ensure that uv is installed. Afterwards it will place agent configuration files
for Claude Code, Codex, GitHub Copilot and Gemini into a subdirectory named
matensemble_campaigns.

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
   cd $SCRATCH/MatEnsemble/matensemble_campaigns

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
-----------------

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

.. code-block:: txt

   Can you use the MatEnsemble MCP server to launch the dashboard and give me the command to access it from localhost?

The agent will launch the dashboard on the login node and provide the command for you to
forward the port to localhost so that you can view your workflows.
