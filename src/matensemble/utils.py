from __future__ import annotations

import json
import threading
from typing import Any

from pathlib import Path
from dataclasses import fields, is_dataclass, replace
from enum import Enum

from collections.abc import Iterable, Mapping

from matensemble.model import OutputReference


def _dashboard_import_error() -> RuntimeError:
    return RuntimeError(
        "The MatEnsemble dashboard requires the optional dashboard dependencies. "
        "Install `starlette` and `uvicorn` in the runtime environment before "
        "calling submit(dashboard=True)."
    )


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, OutputReference):
        return {"__type__": "OutputReference", "chore_id": value.chore_id}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    if isinstance(value, frozenset):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _collect_dep_ids(args, kwargs) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref.chore_id for ref in _find_refs(args, kwargs)))


def _find_refs(args, kwargs):
    """
    Find all OutputReference objects found anywhere inside args or kwargs.
    """

    seen = set()

    def _walk(x):
        obj_id = id(x)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(x, OutputReference):
            yield x
            return

        if isinstance(x, (str, bytes, bytearray)):
            return

        if isinstance(x, Mapping):
            for k, v in x.items():
                yield from _walk(k)
                yield from _walk(v)
            return

        if is_dataclass(x) and not isinstance(x, type):
            for f in fields(x):
                yield from _walk(getattr(x, f.name))
            return

        if isinstance(x, Iterable):
            for item in x:
                yield from _walk(item)
            return

    yield from _walk(args)
    yield from _walk(kwargs)


def _resolve_output_references(value, dep_results):
    """
    Recursively replace OutputReference objects with their concrete dependency
    results, preserving the container/dataclass shape as much as possible.
    """

    if isinstance(value, OutputReference):
        try:
            return dep_results[value.chore_id]
        except KeyError as e:
            raise KeyError(
                f"Missing dependency result for chore_id={value.chore_id!r}"
            ) from e

    if isinstance(value, (str, bytes, bytearray)):
        return value

    if isinstance(value, Mapping):
        resolved = {}
        for k, v in value.items():
            new_k = _resolve_output_references(k, dep_results)
            new_v = _resolve_output_references(v, dep_results)
            resolved[new_k] = new_v
        return type(value)(resolved)

    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            f.name: _resolve_output_references(getattr(value, f.name), dep_results)
            for f in fields(value)
        }
        return replace(value, **updates)

    if isinstance(value, tuple):
        return tuple(_resolve_output_references(v, dep_results) for v in value)

    if isinstance(value, list):
        return [_resolve_output_references(v, dep_results) for v in value]

    if isinstance(value, set):
        return {_resolve_output_references(v, dep_results) for v in value}

    if isinstance(value, frozenset):
        return frozenset(_resolve_output_references(v, dep_results) for v in value)

    return value


def setup_dashboard(status_file: str) -> None:
    app = create_app(status_file)
    try:
        import uvicorn
    except ImportError as exc:
        raise _dashboard_import_error() from exc

    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "warning"},
        daemon=True,
    )

    thread.start()


def create_app(status_file: str) -> Any:
    """
    Create the web app that will run the server which serves the status
    dashboard that users can view on port 8000. Since the workflows will be
    done on a cluster the user will need to ssh tunnel into the allocation
    and port forward port 8000 to their local machine in order to view the server

    Example
    -------
    .. code-block:: bash

        # After launching the chore with dashboard=True note the node and run this command
        ssh -L 8000:frontier00206:8000 kaleb@frontier.olcf.ornl.gov


    """
    try:
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:
        raise _dashboard_import_error() from exc

    status_path = Path(status_file)

    async def get_status(_request):
        try:
            payload = json.loads(status_path.read_text())
        except FileNotFoundError:
            payload = {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "freeCores": 0,
                "freeGpus": 0,
            }
        return JSONResponse(payload)

    base_dir = Path(__file__).resolve().parent
    dist_dir = base_dir / "dash"
    app = Starlette(
        routes=[
            Route("/api/status", get_status),
            Mount("/", StaticFiles(directory=dist_dir, html=True), name="static"),
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
