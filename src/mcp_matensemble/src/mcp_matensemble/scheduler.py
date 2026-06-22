from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .security import append_audit_event, resolve_within_workspace


JOB_ID_RE = re.compile(r"\b(?:Submitted batch job\s+)?(\d+)\b")


def plan_job_submission(script_path: str, *, cwd: Path | None = None) -> dict[str, Any]:
    script = _validate_slurm_script(script_path, cwd=cwd)
    return {
        "script_path": str(script),
        "command": ["sbatch", str(script)],
        "command_text": f"sbatch {script}",
        "execute_default": False,
    }


def submit_job(
    script_path: str,
    *,
    execute: bool = False,
    timeout_seconds: int = 60,
    cwd: Path | None = None,
) -> dict[str, Any]:
    plan = plan_job_submission(script_path, cwd=cwd)
    payload: dict[str, Any] = {
        **plan,
        "executed": False,
    }
    if not execute:
        payload["message"] = "Dry run only. Pass execute=True to submit with sbatch."
        return payload
    if shutil.which("sbatch") is None:
        payload["returncode"] = None
        payload["stdout"] = ""
        payload["stderr"] = "required command not found on PATH: sbatch"
        return payload

    completed = subprocess.run(
        plan["command"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    job_id = _parse_job_id(completed.stdout)
    payload.update(
        {
            "executed": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "job_id": job_id,
        }
    )
    append_audit_event(
        {
            "tool": "submit_matensemble_job",
            "command": plan["command"],
            "returncode": completed.returncode,
            "job_id": job_id,
        },
        cwd=cwd,
    )
    return payload


def get_job_status(job_id: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    if not str(job_id).isdigit():
        raise ValueError("job_id must be numeric")
    if shutil.which("squeue") is None:
        return {
            "job_id": str(job_id),
            "available": False,
            "stdout": "",
            "stderr": "required command not found on PATH: squeue",
        }

    command = [
        "squeue",
        "-j",
        str(job_id),
        "-o",
        "%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "job_id": str(job_id),
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def cancel_job(
    job_id: str,
    *,
    execute: bool = False,
    timeout_seconds: int = 30,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if not str(job_id).isdigit():
        raise ValueError("job_id must be numeric")
    command = ["scancel", str(job_id)]
    payload: dict[str, Any] = {
        "job_id": str(job_id),
        "command": command,
        "executed": False,
    }
    if not execute:
        payload["message"] = "Dry run only. Pass execute=True to cancel with scancel."
        return payload
    if shutil.which("scancel") is None:
        payload["returncode"] = None
        payload["stdout"] = ""
        payload["stderr"] = "required command not found on PATH: scancel"
        return payload

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    payload.update(
        {
            "executed": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    append_audit_event(
        {
            "tool": "cancel_matensemble_job",
            "command": command,
            "returncode": completed.returncode,
            "job_id": str(job_id),
        },
        cwd=cwd,
    )
    return payload


def inspect_outputs(workflow_dir: str, *, cwd: Path | None = None) -> dict[str, Any]:
    root = resolve_within_workspace(workflow_dir, cwd=cwd)
    if not root.is_dir():
        raise ValueError(f"workflow_dir does not exist or is not a directory: {root}")

    status_path = root / "status.json"
    log_path = root / "matensemble_workflow.log"
    out_dir = root / "out"
    chores = []
    if out_dir.is_dir():
        for child in sorted(out_dir.iterdir()):
            if child.is_dir() and child.name.startswith("chore-"):
                chores.append(
                    {
                        "id": child.name,
                        "stdout": str(child / "stdout") if (child / "stdout").exists() else None,
                        "stderr": str(child / "stderr") if (child / "stderr").exists() else None,
                        "metadata": str(child / "metadata.json") if (child / "metadata.json").exists() else None,
                        "result_pickle": str(child / "result.pickle") if (child / "result.pickle").exists() else None,
                    }
                )

    status = None
    status_schema_version = None
    history_path = None
    history = []
    if status_path.exists():
        try:
            import json

            status = json.loads(status_path.read_text(encoding="utf-8"))
            status_schema_version = int(status.get("schema_version", 1))
            history_name = status.get("history_file")
            if status_schema_version == 2 and history_name:
                candidate = root / history_name
                if candidate.is_file():
                    history_path = str(candidate)
                    history = [
                        json.loads(line)
                        for line in candidate.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
        except json.JSONDecodeError as exc:
            status = {"error": f"could not parse status.json: {exc}"}

    return {
        "workflow_dir": str(root),
        "status_path": str(status_path) if status_path.exists() else None,
        "status": status,
        "status_schema_version": status_schema_version,
        "history_path": history_path,
        "history_count": len(history),
        "latest_history": history[-1] if history else None,
        "log_path": str(log_path) if log_path.exists() else None,
        "chore_count": len(chores),
        "chores": chores,
    }


def tail_log(workflow_dir: str, *, lines: int = 100, cwd: Path | None = None) -> dict[str, Any]:
    root = resolve_within_workspace(workflow_dir, cwd=cwd)
    log_path = root / "matensemble_workflow.log"
    if not log_path.exists():
        return {"workflow_dir": str(root), "log_path": str(log_path), "text": ""}
    safe_lines = max(1, min(int(lines), 1000))
    text_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "workflow_dir": str(root),
        "log_path": str(log_path),
        "lines": safe_lines,
        "text": "\n".join(text_lines[-safe_lines:]),
    }


def _validate_slurm_script(script_path: str, *, cwd: Path | None) -> Path:
    script = resolve_within_workspace(script_path, cwd=cwd)
    if script.suffix != ".slurm":
        raise ValueError("script_path must end with .slurm")
    if not script.is_file():
        raise ValueError(f"script_path does not exist: {script}")
    text = script.read_text(encoding="utf-8", errors="replace")
    if "#SBATCH" not in text:
        raise ValueError("script_path does not look like a Slurm script; missing #SBATCH")
    if "matensemble run" not in text and "python " not in text:
        raise ValueError("script_path does not appear to launch a MatEnsemble workflow")
    return script


def _parse_job_id(stdout: str) -> str | None:
    match = JOB_ID_RE.search(stdout)
    return match.group(1) if match else None
