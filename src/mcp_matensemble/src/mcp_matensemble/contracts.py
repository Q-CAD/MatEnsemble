from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .systems import UnsupportedSystemError


def ok(
    result: dict[str, Any] | list[Any] | str | None = None,
    *,
    created_files: list[str] | None = None,
    modified_files: list[str] | None = None,
    deleted_files: list[str] | None = None,
    commands_run: list[list[str]] | None = None,
    commands_not_run: list[list[str]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "result": {} if result is None else result,
        "created_files": created_files or [],
        "modified_files": modified_files or [],
        "deleted_files": deleted_files or [],
        "commands_run": commands_run or [],
        "commands_not_run": commands_not_run or [],
        "warnings": warnings or [],
    }


def error(
    error_code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    commands_not_run: list[list[str]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "created_files": [],
        "modified_files": [],
        "deleted_files": [],
        "commands_run": [],
        "commands_not_run": commands_not_run or [],
        "warnings": warnings or [],
    }


def wrap(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except UnsupportedSystemError as exc:
        return exc.to_error()
    except FileExistsError as exc:
        return error("FILE_EXISTS", str(exc))
    except RuntimeError as exc:
        message = str(exc)
        code = "SCRATCH_REQUIRED" if "$SCRATCH is not set" in message else "RUNTIME_ERROR"
        return error(code, message)
    except ValueError as exc:
        return error("VALIDATION_ERROR", str(exc))
