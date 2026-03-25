"""
Worker entrypoint for executing a serialized MatEnsemble Chore.

This module is meant to be launched inside a Flux job by the manager.
The manager serializes a :obj:`Chore` object to disk
then submits a command to Flux that runs this worker with the chore ID and
path to that chore spec. In practice, the submitted command is conceptually
something like:

.. code-block:: bash

    python -m matensemble.runtime_worker --chore-id <chore_id> --spec-file <path/to/chore.pkl>

When Flux starts that command on an allocated worker, this module:

#. Parses the CLI arguments to determine which chore it is supposed to run
   and where the serialized chore specification lives.

#. Loads the pickled :obj:`Chore` object from disk and validates that the CLI
   chore ID matches the ID stored in the spec file. This provides a basic
   safety check that the correct chore is being executed.

#. Reconstructs the Python import environment needed to run the chore's
   target function. The :obj:`Chore` stores the module path
   and qualified function name. This worker adds the workflow source root to
   ``sys.path``, imports the module with ``importlib.import_module()``,
   and resolves the callable from its qualified name.

#. Loads dependency results for any upstream chores and replaces
   ``OutputReference`` placeholders in ``chore.args`` and ``chore.kwargs``
   with the actual deserialized values. This allows dependent chores to
   receive the concrete outputs of earlier chores.

#. Calls the resolved function as:

   .. code-block:: python

       func(*args, **kwargs)

   where ``args`` and ``kwargs`` are the fully resolved positional and
   keyword arguments from the ``Chore``.

#. Serializes the returned result to ``result.pkl`` for downstream chores
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
            raise ValueError("Chores must reference importable top-level callables")
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
    Takes in the command line arguements and uses them find the :obj:`Chore`, import
    the module where the user defined the funciton, import it then run the funciton
    with the arguments and key-word arguments. Then pickles the result into the outdir.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--chore-id", "--choreid", dest="chore_id", required=True)
    parser.add_argument(
        "--spec-file",
        "--chore-dir",
        "--choredir",
        dest="spec_file",
        required=True,
    )
    ns = parser.parse_args()

    spec_file = Path(ns.spec_file).resolve()

    with spec_file.open("rb") as f:
        chore = pickle.load(f)

    if ns.chore_id != chore.id:
        raise ValueError(
            f"Chore ID mismatch: CLI chore_id={ns.chore_id!r}, spec chore_id={chore.id!r}"
        )

    # spec_file:
    #   <source_root>/matensemble_workflow-.../out/<chore_id>/chore.pkl
    # so source_root is four parents up from the spec file
    source_root = spec_file.parent.parent.parent.parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    module = importlib.import_module(chore.func_module)
    func = _resolve_qualname(module, chore.func_qualname)

    dep_results = {dep: _load_dep_result(spec_file, dep) for dep in chore.deps}
    args = _resolve_output_references(chore.args, dep_results)
    kwargs = _resolve_output_references(chore.kwargs, dep_results)

    result = func(*args, **kwargs)

    with (spec_file.parent / "result.pkl").open("wb") as f:
        pickle.dump(result, f)

    _try_write_result_json(result, spec_file.parent / "result.json")


if __name__ == "__main__":
    main()
