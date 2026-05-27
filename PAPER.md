---
title: 'MatEnsemble: A Flux-native Python workflow manager for adaptive high-throughput computing on HPC systems'
tags:
  - Python
  - high-performance computing
  - high-throughput computing
  - workflow management
  - Flux
  - ensemble simulation
  - materials science
authors:
  - name: Soumendu Bagchi
    affiliation: 1
    # orcid: TODO
  - name: Kaleb Duchesneau
    affiliation: 1
    # orcid: TODO
affiliations:
  - name: TODO: Add institutional affiliation(s)
    index: 1
date: 25 May 2026
bibliography: paper.bib
# repository: TODO: Add repository URL
# archive_doi: TODO: Add Zenodo/software archive DOI after release
---

# Summary

MatEnsemble is a Python package for defining and running high-throughput workflows inside high-performance computing (HPC) allocations. Users construct a directed acyclic graph (DAG) of *chores*, where each chore is either a delayed Python callable or an executable command with explicit resource requirements. MatEnsemble submits these chores through the Flux resource manager, tracks completion, resolves dependencies through serialized Python results, and records workflow state in a structured layout. The package is designed for <insert science cases here> where launching one batch job per task would create excessive scheduler overhead or leave resources idle.

The main user interface is the `Pipeline` object. Decorated Python functions become delayed calls that return `OutputReference` placeholders, passing those placeholders into later chores creates dependency edges. Executable chores can also be added for external programs. At submission time, MatEnsemble validates the dependency graph, submits ready chores that fit the available resources, and continues scheduling until the workflow has no ready, running, or blocked chores. Each run writes logs, standard output and error files for each chore, chore metadata, Python return values, and a status file that can optionally drive a lightweight dashboard.

# Statement of need

Materials science on modern supercomputers increasingly require large volumes of related calculations rather than a single monolithic simulation. Examples include <insert science examples here> These workloads are often awkward to express with traditional batch scripts: a large number of short scheduler submissions can overload shared scheduling infrastructure, while static batching inside a single allocation can waste resources when individual tasks finish at different times. Researchers need a way to keep a single allocation busy while preserving a concise programming model and explicit resource requests for each task.

<!-- TODO: Mention the 100 srun limit and the usefulness of flux and adaptive and maybe user defined here  -->
<!--       Talk about how JobFlow is built on FireWorks manager and the disadvantages of that and how the other managers are python native and the disadvantages -->
<!--       Maybe mention containers??? -->
<!--       Maybe mention How lightweight MatEnsemble is compared to the others??? -->

MatEnsemble addresses this need by combining a Python DAG API with Flux-native execution. The package assumes that the user is already inside a Flux allocation or Flux session, then acts as an internal workflow scheduler for that allocation. Instead of treating the external batch system as the unit of every scientific task, MatEnsemble submits child Flux jobs from a Python driver process and reacts to live resource availability. This design is especially useful for small-to-medium chores where queue latency or fixed waves would dominate useful work. It is also useful for workflows with heterogeneous task sizes because the adaptive scheduler can back-fill the allocation as chores complete.

The target users are computational scientists and HPC developers who need to compose ensembles of Python functions, MPI programs, shell commands, and analysis steps without writing a custom scheduler. The package also targets research software developers who want a lightweight execution layer for Flux-enabled systems while retaining human readable files for debugging and reproducibility.

# State of the field

Several mature Python workflow systems already support scientific task graphs. Parsl provides a broad parallel scripting model for Python functions and external applications across local, cluster, cloud, and grid resources [@babuji2019parsl; @parsl_docs]. Jobflow provides a Pythonic decorator-based workflow model aimed at high-throughput computational workflows, with strong adoption in materials science [@rosen2024jobflow]. libEnsemble focuses on dynamic ensembles using a generator-simulator-allocator model, particularly for adaptive sampling and optimization campaigns [@hudson2025libensemble]. These systems demonstrate the value of Python-native workflows for computational science, and MatEnsemble is complementary rather than a replacement.

MatEnsemble's distinct contribution is its deliberately narrow, Flux-native design. It does not attempt to abstract over every possible execution backend. Instead, it uses Flux as the runtime substrate and focuses on keeping a Flux allocation saturated with dependency-aware chores that request tasks, CPU cores, GPUs, and MPI support. This creates a small conceptual surface area for sites where Flux is already available, while still supporting Python callables, executable commands, dependency resolution, adaptive scheduling, failure propagation, and structured workflow artifacts. Compared with contributing this exact functionality to a larger general-purpose workflow system, a separate package allows MatEnsemble to prioritize Flux-specific resource reporting, site container workflows, and a minimal API tailored to ensemble-style HPC campaigns.

# Software design

MatEnsemble separates workflow construction from execution. During construction, `Pipeline.chore` records delayed Python calls and `Pipeline.exec` records command-line chores. Dependency discovery is data-driven: `OutputReference` objects embedded in positional arguments, keyword arguments, nested containers, or dataclasses are collected into graph edges. Before execution, MatEnsemble builds a `networkx` DAG, rejects missing dependencies and cycles, and orders chores topologically. Each python chore will have its original callable stored in a *registry* and during runtime each chore will have a dedicated directory for all of its outputs.

```
   <basedir or cwd>/
   └── matensemble_workflow-YYYYMMDD_HHMMSS/
       ├── status.json              # Atomically updated for the dashboard / monitoring
       ├── matensemble_workflow.log # Detailed text log from the logger
       └── out/
           ├── registry/            # Pickled chore callables
           │   ├── func_qualname_1
           │   ├── ...
           │   └── func_qualname_n
           ├── <chore_id_1>/
           │   ├── stdout
           │   ├── stderr
           │   ├── metadata.json    # Metadata of the chore in JSON for debugging
           │   ├── chore.pickle     # Pickled chore object
           │   └── result.pickle    # Python chore return value
           ├── ...
           └── <chore_id_n>/
               └── ...
```

At runtime, `FluxManager` owns the ready, blocked, running, completed, and failed chore sets. It obtains a Flux handle, measures currently free cores and GPUs, and repeatedly executes a scheduling loop: refresh resources, write status, submit every ready chore that fits, process completed Flux futures, unblock downstream chores, and repeat. Python chores are launched through `matensemble.runtime_worker`, which reloads a serialized chore specification, loads the registered callable, unpickles upstream results, substitutes them for `OutputReference` placeholders, executes the function, and writes the return value to `result.pickle`. This architecture trades some serialization overhead for a clear boundary between the driver and worker processes, making each chore's command, metadata, stdout, stderr, and result inspectable on disk.

The scheduler uses the *strategy pattern* for future processing. The default adaptive strategy tries to submit newly unblocked work immediately after completions, while the non-adaptive strategy provides a simpler wave-like execution mode. Users can also define their own strategy to be injected into the `FluxManager` at runtime. Users can define a function that takes in the results of a PYTHON `ChoreType` and returns a `ChoreSpec` object. The `FluxManager` will then append that to the end of queue. This effectively allows users to add runtime cycles into their workflow.

# Research impact

MatEnsemble is positioned to support high-throughput computational research on Flux-enabled HPC systems, particularly workflows where many modest tasks must be run inside a single allocation. Community-readiness signals include Sphinx documentation, tutorial workflows, container-oriented installation guidance for HPC systems, a Conda environment, a documented output layout, and explicit runtime status artifacts. The package is also designed around common research needs: MPI chores, GPU requests, Python result passing, failure propagation to dependent tasks, and optional dashboard monitoring. Future submissions should strengthen this section with quantitative benchmarks, named research projects, and publications or reports that used MatEnsemble in production.

# AI usage disclosure

This software and paper was prepared with assistance from many models including Microsoft's Copilot, Anthropic's Claude Sonnet 4.6 and OpenAI's ChatGPT 5.4 & 5.5. The models were used in the initial refactor though the first lines of code were handwritten to establish a structure and design pattern that best fit the scope of the project. After the initial skeleton was established many methods were used to take advantage of the power of these models while ensuring consistency. Methods such as "Spec Driven Development" and "Rubber Duck Debugging". All of the generated code was thoroughly and tests were written to define behavior.

# Acknowledgements

<!-- TODO: Acknowledge funding sources, institutional support, HPC allocations, mentors, and contributors.  -->

# References

[Flux Documentation](https://flux-framework.readthedocs.io/en/latest/)
[JobFlow](https://matgenix.github.io/jobflow-remote/index.html)
[Parsl](https://parsl-project.org/)
[libEnsemble](https://libensemble.readthedocs.io/en/latest/)
<!-- TODO: ADD other sources that were used  -->
