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
        kwargs={"host": "127.0.0.1", "port": 8000, "log_level": "warning"},
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
    status_path = Path(status_file)
    from matensemble.dashboard import WorkflowCatalog, create_dashboard_app

    relative = status_path.parent.name
    if not relative.startswith("matensemble_workflow-"):
        return _create_legacy_fallback_app(status_path)
    root = status_path.parent.parent
    catalog = WorkflowCatalog(root)
    catalog.refresh()
    identifier = next(
        (
            item["id"]
            for item in catalog.catalog()["workflows"]
            if (root / item["relative_path"]) == status_path.parent
        ),
        None,
    )
    if identifier is None:
        return _create_legacy_fallback_app(status_path)
    return create_dashboard_app(
        root,
        compatibility_workflow_id=identifier,
    )


def _create_legacy_fallback_app(status_path: Path) -> Any:
    """Serve old single-workflow routes for non-stamped legacy directories."""
    try:
        from starlette.applications import Starlette
        from starlette.responses import FileResponse, JSONResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise _dashboard_import_error() from exc

    async def get_status(_request):
        from matensemble.logger import normalize_status_payload, read_status

        try:
            return JSONResponse(read_status(status_path))
        except FileNotFoundError:
            return JSONResponse(
                normalize_status_payload(
                    {"state": "initializing"}, status_path=status_path
                )
            )

    async def get_history(_request):
        from matensemble.logger import read_status, read_status_history

        try:
            payload = read_status_history(status_path, read_status(status_path))
        except FileNotFoundError:
            payload = []
        return JSONResponse(payload)

    async def get_stderr(request):
        chore_id = request.path_params["chore_id"]
        if (
            not chore_id.startswith("chore-")
            or "/" in chore_id
            or "\\" in chore_id
            or chore_id in {".", ".."}
        ):
            return JSONResponse({"error": "invalid chore id"}, status_code=400)
        path = status_path.parent / "out" / chore_id / "stderr"
        if not path.is_file():
            return JSONResponse({"error": "stderr not found"}, status_code=404)
        return FileResponse(path, media_type="text/plain")

    return Starlette(
        routes=[
            Route("/api/status", get_status),
            Route("/api/history", get_history),
            Route("/api/artifacts/{chore_id:str}/stderr", get_stderr),
        ]
    )
