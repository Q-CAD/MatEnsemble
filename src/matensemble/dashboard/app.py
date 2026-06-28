from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .discovery import WorkflowCatalog


CHORE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_MAX_POINTS = 1000
MAX_HISTORY_POINTS = 5000


def _error(code: str, message: str, status_code: int):
    from starlette.responses import JSONResponse

    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


def _downsample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    if limit == 1:
        return [records[-1]]
    last = len(records) - 1
    indexes = {round(index * last / (limit - 1)) for index in range(limit)}
    return [records[index] for index in sorted(indexes)]


def read_history(
    workflow_path: Path,
    status: dict[str, Any],
    *,
    after_sequence: int | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    history_name = status.get("history_file")
    if not history_name:
        current = status.get("current", {})
        workflow = status.get("workflow", {})
        records = [
            {
                "sequence": current.get("sequence", 0),
                "timestamp": workflow.get("updated_at"),
                "elapsed_seconds": workflow.get("elapsed_seconds"),
                "state": workflow.get("state"),
                **{
                    key: current.get(key, 0)
                    for key in (
                        "pending",
                        "ready",
                        "blocked",
                        "running",
                        "completed",
                        "failed",
                        "free_cores",
                        "free_gpus",
                    )
                },
            }
        ]
        if after_sequence is not None:
            records = [
                row for row in records if int(row.get("sequence", -1)) > after_sequence
            ]
        return {
            "records": records,
            "first_sequence": records[0]["sequence"] if records else None,
            "last_sequence": records[-1]["sequence"] if records else None,
            "truncated": False,
            "ignored_incomplete_final_line": False,
        }

    if (
        not isinstance(history_name, str)
        or Path(history_name).name != history_name
        or history_name in {".", ".."}
    ):
        raise ValueError("status history_file must be a file name")
    history_path = workflow_path / history_name
    try:
        resolved_history = history_path.resolve()
        resolved_history.relative_to(workflow_path.resolve())
    except (OSError, ValueError):
        raise ValueError("history file resolves outside the workflow directory")
    if not resolved_history.is_file():
        records = []
        ignored = False
    else:
        text = resolved_history.read_text(encoding="utf-8")
        lines = text.splitlines()
        records = []
        ignored = False
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                is_incomplete_final = index == len(lines) - 1 and not text.endswith("\n")
                if is_incomplete_final:
                    ignored = True
                    continue
                raise ValueError(f"invalid JSON history record on line {index + 1}")
            if not isinstance(record, dict):
                raise ValueError(f"history record on line {index + 1} is not an object")
            records.append(record)

    records.sort(key=lambda row: int(row.get("sequence", -1)))
    if after_sequence is not None:
        records = [
            row for row in records if int(row.get("sequence", -1)) > after_sequence
        ]
    truncated = len(records) > max_points
    records = _downsample(records, max_points)
    return {
        "records": records,
        "first_sequence": records[0].get("sequence") if records else None,
        "last_sequence": records[-1].get("sequence") if records else None,
        "truncated": truncated,
        "ignored_incomplete_final_line": ignored,
    }


def create_dashboard_app(
    root: str | Path,
    *,
    scan_interval: float = 5.0,
    stale_after: float = 30.0,
    compatibility_workflow_id: str | None = None,
):
    try:
        from starlette.applications import Starlette
        from starlette.responses import FileResponse, JSONResponse
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "The MatEnsemble dashboard requires starlette and uvicorn."
        ) from exc

    catalog = WorkflowCatalog(root, stale_after=stale_after)
    interval = max(0.1, float(scan_interval))

    async def scanner() -> None:
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(catalog.refresh)

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(catalog.refresh)
        task = asyncio.create_task(scanner())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def get_catalog(_request):
        return JSONResponse(catalog.catalog())

    async def get_status(request):
        identifier = request.path_params["workflow_id"]
        record = await asyncio.to_thread(catalog.status, identifier)
        if record is None:
            return _error(
                "workflow_not_found",
                "The workflow is no longer available.",
                404,
            )
        return JSONResponse(
            {
                "workflow_id": record.id,
                "relative_path": record.relative_path,
                "health": record.health,
                "error": record.error,
                "status": record.status,
            }
        )

    async def get_history(request):
        identifier = request.path_params["workflow_id"]
        record = await asyncio.to_thread(catalog.status, identifier)
        if record is None or record.health == "missing":
            return _error(
                "workflow_not_found",
                "The workflow is no longer available.",
                404,
            )
        if record.status is None:
            return _error(
                "status_unavailable",
                record.error or "Workflow status is not available yet.",
                404,
            )
        try:
            after_raw = request.query_params.get("after_sequence")
            after = int(after_raw) if after_raw is not None else None
            max_points = int(
                request.query_params.get("max_points", DEFAULT_MAX_POINTS)
            )
            if max_points < 1 or max_points > MAX_HISTORY_POINTS:
                raise ValueError
        except ValueError:
            return _error(
                "invalid_history_query",
                f"max_points must be between 1 and {MAX_HISTORY_POINTS}, and "
                "after_sequence must be an integer.",
                400,
            )
        try:
            payload = await asyncio.to_thread(
                read_history,
                record.path,
                record.status,
                after_sequence=after,
                max_points=max_points,
            )
        except OSError as exc:
            return _error(
                "history_unreadable",
                exc.strerror or "The history file could not be read.",
                500,
            )
        except (UnicodeError, ValueError) as exc:
            return _error("history_unreadable", str(exc), 500)
        return JSONResponse({"workflow_id": identifier, **payload})

    async def get_stderr(request):
        identifier = request.path_params["workflow_id"]
        chore_id = request.path_params["chore_id"]
        if not CHORE_ID_RE.fullmatch(chore_id) or chore_id in {".", ".."}:
            return _error("invalid_chore_id", "The chore ID is invalid.", 400)
        record = catalog.get(identifier)
        if record is None or record.health == "missing":
            return _error(
                "workflow_not_found",
                "The workflow is no longer available.",
                404,
            )
        stderr_path = record.path / "out" / chore_id / "stderr"
        try:
            resolved = stderr_path.resolve()
            resolved.relative_to(record.path.resolve())
            resolved.relative_to(catalog.root)
        except (OSError, ValueError):
            return _error(
                "artifact_outside_workflow",
                "The requested artifact is outside the workflow directory.",
                400,
            )
        if not resolved.is_file():
            return _error("artifact_not_found", "stderr was not found.", 404)
        return FileResponse(resolved, media_type="text/plain; charset=utf-8")

    routes = [
        Route("/api/catalog", get_catalog),
        Route("/api/workflows/{workflow_id:str}/status", get_status),
        Route("/api/workflows/{workflow_id:str}/history", get_history),
        Route(
            "/api/workflows/{workflow_id:str}/artifacts/{chore_id:str}/stderr",
            get_stderr,
        ),
    ]

    if compatibility_workflow_id:
        async def legacy_status(request):
            request.path_params["workflow_id"] = compatibility_workflow_id
            response = await get_status(request)
            if response.status_code != 200:
                return response
            payload = json.loads(response.body)
            return JSONResponse(payload["status"] or {})

        async def legacy_history(request):
            request.path_params["workflow_id"] = compatibility_workflow_id
            response = await get_history(request)
            if response.status_code != 200:
                return response
            payload = json.loads(response.body)
            return JSONResponse(payload["records"])

        async def legacy_stderr(request):
            request.path_params["workflow_id"] = compatibility_workflow_id
            return await get_stderr(request)

        routes.extend(
            [
                Route("/api/status", legacy_status),
                Route("/api/history", legacy_history),
                Route("/api/artifacts/{chore_id:str}/stderr", legacy_stderr),
            ]
        )

    static_dir = Path(__file__).resolve().parent / "static"
    routes.append(
        Mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    )
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.catalog = catalog
    return app
