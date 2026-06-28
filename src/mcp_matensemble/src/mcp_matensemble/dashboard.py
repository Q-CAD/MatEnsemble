from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from . import context


def launch_dashboard(
    campaign_root: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict[str, Any]:
    root = _resolve_root(campaign_root)
    host = _validate_host(host)
    port = _validate_port(port)
    pid_path = _pid_path(root, port)
    log_path = _log_path(root, port)
    command = [
        "matensemble",
        "dashboard",
        str(root),
        "--host",
        host,
        "--port",
        str(port),
    ]

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
    running = returncode is None
    return {
        "campaign_root": str(root),
        "host": host,
        "port": port,
        "pid": process.pid,
        "running": running,
        "returncode": returncode,
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "node": socket.gethostname(),
        "remote_url": f"http://{host}:{port}",
        "local_url": f"http://localhost:{port}",
        "command": command,
        "command_text": _quote(command),
        "message": (
            "Dashboard started. Use get_dashboard_access for the SSH tunnel command."
            if running
            else f"Dashboard exited immediately; check {log_path}."
        ),
    }


def get_dashboard_access(
    *,
    login_host: str | None = None,
    login_user: str | None = None,
    system: str | None = None,
    remote_port: int = 8000,
    local_port: int = 8000,
) -> dict[str, Any]:
    remote_port = _validate_port(remote_port)
    local_port = _validate_port(local_port)
    target = _ssh_target(login_host, login_user, system)
    command = [
        "ssh",
        "-N",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        target,
    ]
    return {
        "local_url": f"http://localhost:{local_port}",
        "remote_port": remote_port,
        "local_port": local_port,
        "ssh_target": target,
        "command": command,
        "command_text": _quote(command),
    }


def stop_dashboard(campaign_root: str, *, port: int = 8000) -> dict[str, Any]:
    root = _resolve_root(campaign_root)
    port = _validate_port(port)
    pid_path = _pid_path(root, port)
    pid = _read_pid(pid_path)
    if pid is None:
        return {
            "campaign_root": str(root),
            "port": port,
            "pid_path": str(pid_path),
            "stopped": False,
            "message": "No dashboard PID file was found.",
        }
    if not _pid_running(pid):
        pid_path.unlink(missing_ok=True)
        return {
            "campaign_root": str(root),
            "port": port,
            "pid": pid,
            "pid_path": str(pid_path),
            "stopped": False,
            "message": "Dashboard PID was stale; removed the PID file.",
        }
    os.kill(pid, signal.SIGTERM)
    pid_path.unlink(missing_ok=True)
    return {
        "campaign_root": str(root),
        "port": port,
        "pid": pid,
        "pid_path": str(pid_path),
        "stopped": True,
    }


def _resolve_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if root.is_file() and root.name == "status.json":
        root = root.parent
    if root.name.startswith("matensemble_workflow-") and (root / "status.json").exists():
        root = root.parent
    if not root.is_dir():
        raise ValueError(f"campaign_root does not exist or is not a directory: {root}")
    return root


def _pid_path(root: Path, port: int) -> Path:
    return root / f"matensemble-dashboard-{port}.pid"


def _log_path(root: Path, port: int) -> Path:
    return root / f"matensemble-dashboard-{port}.log"


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdigit() else None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _validate_port(value: int) -> int:
    port = int(value)
    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535")
    return port


def _validate_host(value: str) -> str:
    host = value.strip()
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("host must be 127.0.0.1 or localhost")
    return host


def _ssh_target(
    login_host: str | None,
    login_user: str | None,
    system: str | None = None,
) -> str:
    host = _normalize_login_host(login_host, system)
    user = login_user.strip() if login_user else _default_login_user(system)
    return f"{user}@{host}" if user else host


def _normalize_login_host(login_host: str | None, system: str | None) -> str:
    normalized_system = context.normalize_system(system) if system is not None else None
    host = login_host.strip() if login_host else _default_login_host(normalized_system)
    if not host:
        return "<login.host>"
    if normalized_system is None:
        return host

    if normalized_system == "frontier":
        if "." not in host:
            return f"{host}.frontier.olcf.ornl.gov"
        if host.endswith(".frontier"):
            return f"{host}.olcf.ornl.gov"
    return host


def _default_login_host(normalized_system: str | None) -> str:
    if normalized_system == "perlmutter":
        return "perlmutter.nersc.gov"
    if normalized_system == "pathfinder":
        return "pflogin.ornl.gov"
    return socket.gethostname()


def _default_login_user(system: str | None) -> str:
    if system is None:
        return ""
    return os.environ.get("USER") or os.environ.get("LOGNAME") or ""


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
