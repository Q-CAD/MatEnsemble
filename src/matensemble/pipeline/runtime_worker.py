from __future__ import annotations

import argparse
import importlib
import json
import pickle
from pathlib import Path
from typing import Any

from .model import ArgSpec


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
    module = spec["python"]["module"]
    qualname = spec["python"]["qualname"]

    if not module or not qualname:
        raise RuntimeError("Python node spec missing module/qualname")

    mod = importlib.import_module(module)
    fn = _resolve_qualname(mod, qualname)

    pos_args = [_resolve_arg(a, workdir) for a in spec.get("args", [])]
    kw_args = {k: _resolve_arg(v, workdir) for k, v in spec.get("kwargs", {}).items()}

    result = fn(*pos_args, **kw_args)

    _dump_pickle(workdir / "result.pkl", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
