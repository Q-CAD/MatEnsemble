from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .security import append_audit_event, resolve_within_workspace


def plan_dashboard_access(
    dashboard_root: str,
    *,
    login_host: str | None = None,
    login_user: str | None = None,
    host: str = "127.0.0.1",
    remote_port: int = 8000,
    local_port: int = 8000,
    cwd: Path | None = None,
) -> dict[str, Any]:
    root = _resolve_dashboard_root(dashboard_root, cwd=cwd)
    remote_port = _validate_port(remote_port, "remote_port")
    local_port = _validate_port(local_port, "local_port")
    host = _validate_host(host)
    ssh_target = _ssh_target(login_host, login_user)
    start_command = _background_start_command(root, host=host, port=remote_port)
    forward_command = [
        "ssh",
        "-N",
        "-L",
        f"{local_port}:{host}:{remote_port}",
        ssh_target,
    ]

    return {
        "dashboard_root": str(root),
        "host": host,
        "remote_port": remote_port,
        "local_port": local_port,
        "url": f"http://localhost:{local_port}",
        "start_command": start_command,
        "start_command_text": " ".join(shlex.quote(part) for part in start_command),
        "background_start_command": _background_start_command_text(
            root, host=host, port=remote_port
        ),
        "forward_command": forward_command,
        "forward_command_text": " ".join(
            shlex.quote(part) for part in forward_command
        ),
        "stop_command_text": (
            f"kill $(cat {shlex.quote(str(_pid_path(root, remote_port)))})"
        ),
        "notes": [
            "Run the dashboard command on the login node that can read the "
            "shared campaign directory.",
            "Run the SSH forwarding command from your laptop and keep it open "
            "while viewing the dashboard.",
            "Open the URL in a local browser after the tunnel is connected.",
        ],
    }


def start_dashboard(
    dashboard_root: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    execute: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    root = _resolve_dashboard_root(dashboard_root, cwd=cwd)
    port = _validate_port(port, "port")
    host = _validate_host(host)
    command = _start_command(root, host=host, port=port)
    pid_path = _pid_path(root, port)
    log_path = _log_path(root, port)
    payload: dict[str, Any] = {
        "dashboard_root": str(root),
        "host": host,
        "port": port,
        "url_on_login_node": f"http://{host}:{port}",
        "command": command,
        "command_text": " ".join(shlex.quote(part) for part in command),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "executed": False,
    }
    if not execute:
        payload["message"] = "Dry run only. Pass execute=True to start the dashboard."
        return payload
    if not _command_available(command):
        payload["returncode"] = None
        payload["stdout"] = ""
        payload["stderr"] = (
            "required command not found on PATH and Python module fallback is not "
            "available: matensemble"
        )
        return payload
    if not _port_available(host, port):
        payload["returncode"] = None
        payload["stdout"] = ""
        payload["stderr"] = f"{host}:{port} is already in use"
        return payload

    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(0.2)
    returncode = process.poll()
    payload.update(
        {
            "executed": True,
            "pid": process.pid,
            "returncode": returncode,
            "running": returncode is None,
            "message": (
                "Dashboard started. Forward the port from your laptop to view it."
                if returncode is None
                else f"Dashboard process exited immediately; check {log_path}."
            ),
        }
    )
    append_audit_event(
        {
            "tool": "start_matensemble_dashboard",
            "command": command,
            "pid": process.pid,
            "returncode": returncode,
            "dashboard_root": str(root),
            "port": port,
        },
        cwd=cwd,
    )
    return payload


def get_dashboard_status(
    dashboard_root: str,
    *,
    port: int = 8000,
    cwd: Path | None = None,
) -> dict[str, Any]:
    root = _resolve_dashboard_root(dashboard_root, cwd=cwd)
    port = _validate_port(port, "port")
    pid_path = _pid_path(root, port)
    pid = _read_pid(pid_path)
    running = _pid_running(pid) if pid is not None else False
    return {
        "dashboard_root": str(root),
        "port": port,
        "pid_path": str(pid_path),
        "log_path": str(_log_path(root, port)),
        "pid": pid,
        "running": running,
    }


def stop_dashboard(
    dashboard_root: str,
    *,
    port: int = 8000,
    cwd: Path | None = None,
) -> dict[str, Any]:
    root = _resolve_dashboard_root(dashboard_root, cwd=cwd)
    port = _validate_port(port, "port")
    pid_path = _pid_path(root, port)
    pid = _read_pid(pid_path)
    if pid is None:
        return {
            "dashboard_root": str(root),
            "port": port,
            "pid_path": str(pid_path),
            "stopped": False,
            "message": "No dashboard PID file was found.",
        }
    if not _pid_running(pid):
        pid_path.unlink(missing_ok=True)
        return {
            "dashboard_root": str(root),
            "port": port,
            "pid_path": str(pid_path),
            "pid": pid,
            "stopped": False,
            "message": "Dashboard PID was stale; removed the PID file.",
        }

    os.kill(pid, signal.SIGTERM)
    pid_path.unlink(missing_ok=True)
    append_audit_event(
        {
            "tool": "stop_matensemble_dashboard",
            "pid": pid,
            "dashboard_root": str(root),
            "port": port,
        },
        cwd=cwd,
    )
    return {
        "dashboard_root": str(root),
        "port": port,
        "pid_path": str(pid_path),
        "pid": pid,
        "stopped": True,
    }


def _resolve_dashboard_root(path: str, *, cwd: Path | None) -> Path:
    resolved = resolve_within_workspace(path, cwd=cwd)
    if resolved.is_file():
        if resolved.name != "status.json":
            raise ValueError("dashboard_root must be a directory or a status.json file")
        resolved = resolved.parent
    if not resolved.is_dir():
        raise ValueError(
            f"dashboard_root does not exist or is not a directory: {resolved}"
        )
    if (
        resolved.name.startswith("matensemble_workflow-")
        and (resolved / "status.json").exists()
    ):
        return resolved.parent
    return resolved


def _start_command(root: Path, *, host: str, port: int) -> list[str]:
    if shutil.which("matensemble") is not None:
        return [
            "matensemble",
            "dashboard",
            str(root),
            "--host",
            host,
            "--port",
            str(port),
        ]
    return [
        sys.executable,
        "-m",
        "matensemble",
        "dashboard",
        str(root),
        "--host",
        host,
        "--port",
        str(port),
    ]


def _background_start_command(root: Path, *, host: str, port: int) -> list[str]:
    return _start_command(root, host=host, port=port)


def _background_start_command_text(root: Path, *, host: str, port: int) -> str:
    start_command = _background_start_command(root, host=host, port=port)
    command = " ".join(shlex.quote(part) for part in start_command)
    log_path = shlex.quote(str(_log_path(root, port)))
    pid_path = shlex.quote(str(_pid_path(root, port)))
    return f"nohup {command} > {log_path} 2>&1 & echo $! > {pid_path}"


def _pid_path(root: Path, port: int) -> Path:
    return root / f"matensemble-dashboard-{port}.pid"


def _log_path(root: Path, port: int) -> Path:
    return root / f"matensemble-dashboard-{port}.log"


def _read_pid(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(value) if value.isdigit() else None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _command_available(command: list[str]) -> bool:
    if command[0] == sys.executable:
        try:
            import matensemble  # noqa: F401
        except ImportError:
            return False
        return True
    if shutil.which(command[0]) is not None:
        return True
    return False


def _validate_port(value: int, name: str) -> int:
    port = int(value)
    if port < 1 or port > 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


def _validate_host(value: str) -> str:
    host = value.strip()
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("host must be 127.0.0.1 or localhost")
    return host


def _ssh_target(login_host: str | None, login_user: str | None) -> str:
    host = login_host.strip() if login_host else "<login.host>"
    user = login_user.strip() if login_user else None
    if not user:
        return host
    return f"{user}@{host}"
