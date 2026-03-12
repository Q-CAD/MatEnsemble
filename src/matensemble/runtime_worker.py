# runtime_worker.py
import argparse
import importlib
import json
import pickle

from pathlib import Path

from matensemble.job import Job
from matensemble.utils import _json_safe, _resolve_output_references


def _resolve_qualname(module, qualname: str):
    obj = module
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError("Jobs must reference importable top-level callables")
        obj = getattr(obj, part)
    return obj


def _load_dep_result(spec_file: Path, dep_id: str):
    dep_result = spec_file.parent.parent / dep_id / "result.pkl"
    with dep_result.open("rb") as f:
        return pickle.load(f)


def _try_write_result_json(result, out_file):
    try:
        with out_file.open("w") as f:
            json.dump(_json_safe(result), f, indent=2)
    except Exception:
        pass


def main():
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

    spec_file = Path(ns.spec_file)

    with spec_file.open("rb") as f:
        job = pickle.load(f)

    if ns.job_id != job.id:
        raise ValueError(
            f"Job ID mismatch: CLI job_id={ns.job_id!r}, spec job_id={job.id!r}"
        )

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
