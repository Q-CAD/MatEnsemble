.. MatEnsemble documentation master file, created by
   sphinx-installation on Thu Feb 12 16:35:41 2026.

MatEnsemble Documentation
=========================

MatEnsemble is a framework to build, orchestrate, and asynchronously manage extremely scalable adaptive-learning workflows, especially targeted for compute-intensive AI-driven high-throughput and ensemble-driven materials modeling simulations (e.g., atomistic modeling, Phase-Field, etc.) as efficiently as possible. 

While it can in general run on your personal Mac/Linux workstation and orchestrate any python callable, shell commands with explicit resource and dependency-aware execution graphs from a single python workflow driver process,  MatEnsemble shines with ***user-defined autonomous strategic*** execution of large batches of adaptively and hierarchically-scheduled tasks on HPC systems, often on Peta and Exascale computing facilities, e.g., Perlmutter, Frontier, Aurora etc.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   overview
   installation
   mcp
   examples
   tutorials
   design
   reference
   team

.. _api-reference:

.. toctree::
   :maxdepth: 3
   :caption: API

   api/modules
