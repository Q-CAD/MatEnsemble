"""
Worker entrypoint for executing a serialized MatEnsemble Job.

This module is meant to be launched inside a Flux job by the manager.
The manager serializes a :obj:`Job` object to disk
then submits a command to Flux that runs this worker with the job ID and
path to that job spec. In practice, the submitted command is conceptually
something like:

.. code-block:: bash

    python -m matensemble.runtime_worker --job-id <job_id> --spec-file <path/to/job.pkl>

When Flux starts that command on an allocated worker, this module:

#. Parses the CLI arguments to determine which job it is supposed to run
   and where the serialized job specification lives.

#. Loads the pickled :obj:`Job` object from disk and validates that the CLI
   job ID matches the ID stored in the spec file. This provides a basic
   safety check that the correct job is being executed.

#. Reconstructs the Python import environment needed to run the job's
   target function. The :obj:`Job` stores the module path
   and qualified function name. This worker adds the workflow source root to
   ``sys.path``, imports the module with ``importlib.import_module()``,
   and resolves the callable from its qualified name.

#. Loads dependency results for any upstream jobs and replaces
   ``OutputReference`` placeholders in ``job.args`` and ``job.kwargs``
   with the actual deserialized values. This allows dependent jobs to
   receive the concrete outputs of earlier jobs.

#. Calls the resolved function as:

   .. code-block:: python

       func(*args, **kwargs)

   where ``args`` and ``kwargs`` are the fully resolved positional and
   keyword arguments from the ``Job``.

#. Serializes the returned result to ``result.pkl`` for downstream jobs
   and also attempts to write a JSON-friendly version to ``result.json``
   for easier debugging and inspection.

"""

import argparse
import importlib
import json
import inspect
import pickle
import sys

from pathlib import Path

from matensemble.utils import _json_safe, _resolve_output_references


def _resolve_qualname(module, qualname: str):
    """
    Gets the function name from the module
    """

    obj = module
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError("Jobs must reference importable top-level callables")
        obj = getattr(obj, part)
    return inspect.unwrap(obj)


def _load_dep_result(spec_file: Path, dep_id: str):
    """
    Loads the results of the dependencies and returns them
    """

    dep_result = spec_file.parent.parent / dep_id / "result.pkl"
    with dep_result.open("rb") as f:
        return pickle.load(f)


def _try_write_result_json(result, out_file):
    """
    Tries to write a human readable version of the result
    """

    try:
        with out_file.open("w") as f:
            json.dump(_json_safe(result), f, indent=2)
    except Exception:
        pass


def main():
    """
    Takes in the command line arguements and uses them find the :obj:`Job`, import
    the module where the user defined the funciton, import it then run the funciton
    with the arguments and key-word arguments. Then pickles the result into the outdir.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "--jobid", dest="job_id", required=True)
    parser.add_argument(
        "--spec-file",
        "--job-dir",
        "--jobdir",
        dest="spec_file",
        required=True,
    )
    ns = parser.parse_args()

    spec_file = Path(ns.spec_file).resolve()

    with spec_file.open("rb") as f:
        job = pickle.load(f)

    if ns.job_id != job.id:
        raise ValueError(
            f"Job ID mismatch: CLI job_id={ns.job_id!r}, spec job_id={job.id!r}"
        )

    # spec_file:
    #   <source_root>/matensemble_workflow-.../out/<job_id>/job.pkl
    # so source_root is four parents up from the spec file
    source_root = spec_file.parent.parent.parent.parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    module = importlib.import_module(job.func_module)
    func = _resolve_qualname(module, job.func_qualname)

    dep_results = {dep: _load_dep_result(spec_file, dep) for dep in job.deps}
    args = _resolve_output_references(job.args, dep_results)
    kwargs = _resolve_output_references(job.kwargs, dep_results)

    result = func(*args, **kwargs)

    with (spec_file.parent / "result.pkl").open("wb") as f:
        pickle.dump(result, f)

    _try_write_result_json(result, spec_file.parent / "result.json")


if __name__ == "__main__":
    main()
