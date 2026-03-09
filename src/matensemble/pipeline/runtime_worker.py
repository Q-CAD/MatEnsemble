# runtime_worker.py
import argparse
import importlib
import pickle
from pathlib import Path

from matensemble.pipeline.pipeline import Job, OutputReference


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


def _resolve_refs(value, dep_results):
    if isinstance(value, OutputReference):
        return dep_results[value.job_id]
    if isinstance(value, tuple):
        return tuple(_resolve_refs(v, dep_results) for v in value)
    if isinstance(value, list):
        return [_resolve_refs(v, dep_results) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, dep_results) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "--jobid", dest="job_id", required=True)
    parser.add_argument("--job-dir", "--jobdir", dest="job_dir", required=True)
    ns = parser.parse_args()

    spec_file = Path(ns.job_dir)

    with spec_file.open("rb") as f:
        job: Job = pickle.load(f)

    module = importlib.import_module(job.func_module)
    func = _resolve_qualname(module, job.func_qualname)

    dep_results = {dep: _load_dep_result(spec_file, dep) for dep in job.deps}
    args = _resolve_refs(job.args, dep_results)
    kwargs = _resolve_refs(job.kwargs, dep_results)

    result = func(*args, **kwargs)

    with (spec_file.parent / "result.pkl").open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
