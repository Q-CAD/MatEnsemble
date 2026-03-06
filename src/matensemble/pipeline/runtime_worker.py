from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pickle
import sys
import hashlib
from pathlib import Path
from typing import Any


def _load_node_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_qualname(mod: Any, qualname: str) -> Any:
    """
    Resolve a dotted qualname against a module object.
    """
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _load_module_from_path(source_path: str) -> Any:
    """Load a Python module from a file path under a stable synthetic name."""
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"source_path does not exist: {source_path}")
    name = "matensemble_user_" + hashlib.sha1(str(p).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {source_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _dump_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(value, f)


def _resolve_arg(arg: dict[str, Any], node_workdir: Path) -> Any:
    """
    Resolve a serialized ArgSpec into a concrete Python value.

    MVP rules:
    - lit: return value directly
    - node_result: load <out_dir>/<upstream_id>/result.pkl by walking from base out dir
      (we infer base out_dir as node_workdir.parent)
    - output: return absolute path to <out_dir>/<upstream_id>/<relpath>
    """
    t = arg["type"]
    if t == "lit":
        return arg["value"]

    out_dir = node_workdir.parent

    if t == "node_result":
        upstream = arg["node"]
        return _load_pickle(out_dir / upstream / "result.pkl")

    if t == "output":
        upstream = arg["node"]
        rel = arg[
            "value"
        ]  # optional usage; for now store relpath in value or key mapping
        key = arg.get("key")
        # For MVP, store relpath in "value" OR reconstruct via declared outputs in spec.
        # This can be refined once node_spec includes outputs_declared.
        if rel is None:
            raise ValueError("output ArgSpec missing relpath")
        return str(out_dir / upstream / rel)

    raise ValueError(f"Unknown ArgSpec type: {t}")


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for Flux-executed Python tasks.

    Reads node spec, imports user function, resolves inputs, runs the function,
    stores result in workdir/result.pkl, and exits.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--node-spec", required=True)
    args = p.parse_args(argv)

    node_spec_path = Path(args.node_spec)
    spec = _load_node_spec(node_spec_path)

    workdir = node_spec_path.parent
    py = spec.get("python", {})
    module = py.get("module")
    qualname = py.get("qualname")
    source_path = py.get("source_path")

    if not qualname:
        raise RuntimeError("Python node spec missing qualname")

    if "<locals>" in qualname:
        raise RuntimeError(
            "MatEnsemble tasks must be top-level (no nested defs/lambdas). "
            f"Got qualname={qualname!r}"
        )

    # Prefer importing by module name when possible, but fall back to loading
    # from an explicit source path (especially when tasks were defined in __main__).
    mod: Any | None = None
    if module and module != "__main__":
        try:
            mod = importlib.import_module(module)
        except Exception:
            mod = None

    if mod is None:
        if not source_path:
            raise RuntimeError(
                "Cannot import task function: module import failed and no source_path "
                "was recorded. Put tasks in an importable module, or ensure the defining "
                ".py file is available and that pipeline.run(...) is guarded by "
                "if __name__ == '__main__':"
            )
        mod = _load_module_from_path(source_path)

    fn = _resolve_qualname(mod, qualname)

    pos_args = [_resolve_arg(a, workdir) for a in spec.get("args", [])]
    kw_args = {k: _resolve_arg(v, workdir) for k, v in spec.get("kwargs", {}).items()}

    result = fn(*pos_args, **kw_args)

    _dump_pickle(workdir / "result.pkl", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
