# Specification: MatEnsemble MCP Server

## 1. Overview

### Project Name
MatEnsemble MCP Server

### Purpose
Provide a Model Context Protocol server that allows researchers and scientists to define, run, monitor, and analyze materials science workflows using the MatEnsemble Python API.

The server should allow users to interact with MatEnsemble from MCP-compatible clients such as Cursor, VS Code, Claude Desktop, and other agentic tools.

### Target Users
Researchers and scientists at ORNL who run materials science calculations on local workstations or HPC systems such as Frontier.

### Primary Goals
- Let users create MatEnsemble workflows through natural language.
- Let users define and compose MatEnsemble `Chores`.
- Support workflow dependencies between Chores.
- Submit workflows to a cluster or scheduler-backed execution environment.
- Monitor running, completed, and failed workflows.
- Retrieve logs, outputs, artifacts, and calculation summaries.
- Provide safe guardrails for HPC execution.

---

## 2. Background

MatEnsemble provides a Python API for defining units of work called `Chores`. A `Chore` represents a single computational task or workflow step. Chores may depend on other Chores and are scheduled by MatEnsemble according to dependency constraints.

Example concepts:

```python
from matensemble import Workflow, Chore

relax = Chore(
    name="relax_structure",
    command="python relax.py",
    inputs=["POSCAR"],
    outputs=["CONTCAR"]
)

dos = Chore(
    name="compute_dos",
    command="python dos.py",
    inputs=["CONTCAR"],
    outputs=["DOSCAR"],
    depends_on=[relax]
)

workflow = Workflow(name="vasp_relax_dos")
workflow.add(relax)
workflow.add(dos)
workflow.run()
