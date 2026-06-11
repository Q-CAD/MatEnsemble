from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import shutil
import subprocess
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import error, ok
from .environment import resolve_image_tag
from .examples import (
    get_container_template as read_container_template,
    get_system_example,
    list_container_templates as discover_container_templates,
    list_examples_for_system,
)
from .scriptgen import VALID_WORKFLOW_KINDS, render_workflow_script
from .security import (
    ensure_safe_generated_text,
    relative_to_workspace,
    resolve_within_workspace,
    scan_dangerous_text,
    scratch_workspace_root,
    slugify,
    validate_campaign_name,
)
from .systems import SUPPORTED_SYSTEMS, get_system_profile, list_systems, normalize_system

ALLOWED_COMMANDS = {
    "python",
    "uv",
    "apptainer",
    "podman-hpc",
    "docker",
    "sbatch",
    "squeue",
    "scancel",
    "matensemble",
    "flux",
    "chmod",
    "mkdir",
    "cp",
}

PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.@+:-]+$")
JOB_ID_RE = re.compile(r"\bSubmitted batch job\s+(\d+)\b")
SOURCE_SYMBOLS = {
    "pipeline": "src/matensemble/pipeline.py",
    "manager": "src/matensemble/manager.py",
    "fluxlet": "src/matensemble/fluxlet.py",
    "strategy": "src/matensemble/strategy.py",
    "runtime_worker": "src/matensemble/runtime_worker.py",
    "model": "src/matensemble/model.py",
    "chore": "src/matensemble/chore.py",
    "utils": "src/matensemble/utils.py",
}


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "matensemble").is_dir() and (parent / "example_workflows").is_dir():
            return parent
    raise RuntimeError("MatEnsemble repository root could not be located")


def list_supported_systems() -> dict[str, Any]:
    return ok({"supported_systems": list(SUPPORTED_SYSTEMS), "systems": list_systems()})


def get_system_context(system: str) -> dict[str, Any]:
    key = normalize_system(system)
    profile = get_system_profile(key)
    root = repo_root()
    docs = [
        str(path.relative_to(root))
        for path in sorted((root / "docs" / "source").glob("*.rst"))
    ]
    result = {
        **profile.to_dict(),
        "docs": docs,
        "examples": list_examples_for_system(key),
        "container_templates": discover_container_templates(key),
        "workspace_root": str(scratch_workspace_root()),
    }
    return ok(result)


def list_examples(system: str | None = None) -> dict[str, Any]:
    return ok({"examples": list_examples_for_system(system)})


def get_example(system: str, name: str) -> dict[str, Any]:
    return ok({"system": normalize_system(system), "name": name, "text": get_system_example(system, name)})


def list_container_templates(system: str | None = None) -> dict[str, Any]:
    return ok({"templates": discover_container_templates(system)})


def get_container_template(system: str, filename: str) -> dict[str, Any]:
    return ok(
        {
            "system": normalize_system(system),
            "filename": filename,
            "text": read_container_template(system, filename),
        }
    )


def explain_matensemble_api(symbol: str) -> dict[str, Any]:
    root = repo_root()
    query = symbol.strip()
    if not query:
        raise ValueError("symbol is required")
    lower = query.lower()
    candidate_files = []
    if lower in SOURCE_SYMBOLS:
        candidate_files.append(root / SOURCE_SYMBOLS[lower])
    candidate_files.extend(root / path for path in SOURCE_SYMBOLS.values())

    seen: set[Path] = set()
    matches: list[dict[str, Any]] = []
    for path in candidate_files:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = ast.parse(text)
        lines = text.splitlines()
        for node in ast.walk(parsed):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if query not in node.name and query not in path.stem:
                continue
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            snippet = "\n".join(lines[start - 1 : min(end, start + 40)])
            matches.append(
                {
                    "symbol": node.name,
                    "kind": type(node).__name__.replace("Def", "").lower(),
                    "file": str(path.relative_to(root)),
                    "start_line": start,
                    "end_line": end,
                    "docstring": ast.get_docstring(node),
                    "snippet": snippet,
                }
            )
        if not matches and (lower == path.stem or query in text):
            line_no = next(
                (idx for idx, line in enumerate(lines, start=1) if query in line),
                1,
            )
            snippet = "\n".join(lines[max(0, line_no - 4) : line_no + 16])
            matches.append(
                {
                    "symbol": path.stem,
                    "kind": "module",
                    "file": str(path.relative_to(root)),
                    "start_line": max(1, line_no - 3),
                    "end_line": min(len(lines), line_no + 16),
                    "docstring": ast.get_docstring(parsed),
                    "snippet": snippet,
                }
            )

    related_examples = [
        example
        for example in list_examples_for_system()
        if lower in str(example.get("name", "")).lower()
        or lower in str(example.get("demonstrates", "")).lower()
    ][:5]
    related_tests = [
        str(path.relative_to(root))
        for path in sorted((root / "tests").glob(f"*{lower}*.py"))
    ][:5]
    return ok(
        {
            "query": query,
            "matches": matches[:8],
            "related_examples": related_examples,
            "related_tests": related_tests,
            "notes": [
                "Use Pipeline for workflow construction.",
                "Prefer argv lists for executable chores.",
                "Launch only through prepared plans and confirmation tools.",
            ],
        }
    )


def create_campaign(
    campaign_name: str,
    system: str,
    *,
    auto_suffix: bool = False,
    overwrite: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    raw_name = campaign_name.strip()
    if Path(raw_name).is_absolute() or ".." in Path(raw_name).parts or "/" in raw_name or "\\" in raw_name:
        raise ValueError(f"campaign_name must be a simple directory name: {campaign_name}")
    name = validate_campaign_name(slugify(campaign_name))
    workspace = scratch_workspace_root(cwd=cwd)
    workspace.mkdir(parents=True, exist_ok=True)
    base = f"{name}_{key}_{date.today().isoformat()}"
    campaign_dir = workspace / base
    if campaign_dir.exists() and not overwrite:
        if not auto_suffix:
            raise FileExistsError(f"campaign directory already exists: {campaign_dir}")
        index = 2
        while (workspace / f"{base}_{index:03d}").exists():
            index += 1
        campaign_dir = workspace / f"{base}_{index:03d}"
    campaign_dir.mkdir(parents=True, exist_ok=overwrite)
    for child in ("launch_logs",):
        (campaign_dir / child).mkdir(exist_ok=True)
    metadata = {
        "campaign_name": name,
        "system": key,
        "created_at": _now(),
        "campaign_dir": str(campaign_dir),
    }
    metadata_path = campaign_dir / "mcp_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ok(
        {
            "campaign": campaign_dir.name,
            "campaign_dir": str(campaign_dir),
            "relative_path": str(campaign_dir.relative_to(workspace)),
            "metadata": metadata,
        },
        created_files=[relative_to_workspace(metadata_path, cwd=cwd)],
    )


def list_campaigns(*, cwd: Path | None = None) -> dict[str, Any]:
    workspace = scratch_workspace_root(cwd=cwd)
    workspace.mkdir(parents=True, exist_ok=True)
    campaigns = []
    for path in sorted(workspace.iterdir()):
        if path.is_dir() and path.name not in {"containers", ".vscode"}:
            campaigns.append(
                {
                    "campaign": path.name,
                    "campaign_dir": str(path),
                    "has_workflow": (path / "workflow.py").is_file(),
                    "has_batch_script": (path / "submit.slurm").is_file(),
                    "has_launch_plan": (path / "launch_plan.json").is_file(),
                }
            )
    return ok({"campaigns": campaigns})


def get_campaign_status(campaign: str, *, cwd: Path | None = None) -> dict[str, Any]:
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    files = {
        "workflow": campaign_dir / "workflow.py",
        "batch_script": campaign_dir / "submit.slurm",
        "interactive_script": campaign_dir / "run_interactive.sh",
        "launch_plan": campaign_dir / "launch_plan.json",
        "metadata": campaign_dir / "mcp_metadata.json",
    }
    return ok(
        {
            "campaign": campaign_dir.name,
            "campaign_dir": str(campaign_dir),
            "files": {name: str(path) if path.exists() else None for name, path in files.items()},
        }
    )


def write_workflow(
    campaign: str,
    science_goal: str,
    *,
    workflow_kind: str = "python_dag",
    overwrite: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if workflow_kind not in VALID_WORKFLOW_KINDS:
        raise ValueError(f"workflow_kind must be one of {', '.join(VALID_WORKFLOW_KINDS)}")
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    target = campaign_dir / "workflow.py"
    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(f"workflow.py already exists: {target}")
    text = render_workflow_script(
        campaign_name=campaign_dir.name,
        science_goal=science_goal,
        workflow_kind=workflow_kind,
    )
    _validate_workflow_text(text)
    target.write_text(text, encoding="utf-8")
    rel = relative_to_workspace(target, cwd=cwd)
    return ok({"path": rel}, created_files=[] if existed else [rel], modified_files=[rel] if existed else [])


def write_batch_script(
    campaign: str,
    system: str,
    *,
    account: str | None,
    nodes: int = 1,
    walltime: str = "00:30:00",
    queue: str = "batch",
    gpus: int = 0,
    tasks: int = 1,
    container_backend: str = "apptainer",
    container_path: str | None = None,
    overwrite: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    if not account:
        return error("ACCOUNT_REQUIRED", "Slurm account/project allocation is required.")
    profile = get_system_profile(key)
    if container_backend not in profile.container_backends:
        raise ValueError(f"container_backend must be one of {profile.container_backends}")
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    workflow = campaign_dir / "workflow.py"
    if not workflow.is_file():
        raise ValueError("campaign must contain workflow.py before writing a batch script")
    target = campaign_dir / "submit.slurm"
    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(f"submit.slurm already exists: {target}")
    container = container_path or f"../containers/{key}/matensemble.sif"
    text = _render_batch(
        system=key,
        campaign=campaign_dir.name,
        account=account,
        nodes=nodes,
        walltime=walltime,
        queue=queue,
        gpus=gpus,
        tasks=tasks,
        container_backend=container_backend,
        container_path=container,
    )
    ensure_safe_generated_text(text)
    target.write_text(text, encoding="utf-8")
    rel = relative_to_workspace(target, cwd=cwd)
    return ok(
        {"path": rel, "resources": _resources(account, nodes, walltime, queue, gpus, tasks)},
        created_files=[] if existed else [rel],
        modified_files=[rel] if existed else [],
    )


def validate_campaign(campaign: str, *, cwd: Path | None = None) -> dict[str, Any]:
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    warnings: list[str] = []
    workflow = campaign_dir / "workflow.py"
    batch = campaign_dir / "submit.slurm"
    if not workflow.is_file():
        return error("MISSING_WORKFLOW", "campaign is missing workflow.py")
    text = workflow.read_text(encoding="utf-8", errors="replace")
    try:
        _validate_workflow_text(text)
        py_compile.compile(str(workflow), doraise=True)
    except (ValueError, py_compile.PyCompileError) as exc:
        return error("INVALID_WORKFLOW", str(exc))
    if batch.is_file():
        dangerous = scan_dangerous_text(batch.read_text(encoding="utf-8", errors="replace"))
        if dangerous:
            return error("DANGEROUS_SCRIPT", "batch script contains forbidden patterns", details={"patterns": dangerous})
    else:
        warnings.append("campaign is missing submit.slurm")
    return ok({"campaign": campaign_dir.name, "valid": True}, warnings=warnings)


def prepare_launch_plan(
    campaign: str,
    system: str,
    *,
    mode: str = "batch",
    account: str | None = None,
    nodes: int = 1,
    walltime: str = "00:30:00",
    queue: str = "batch",
    gpus: int = 0,
    tasks: int = 1,
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    if mode not in {"batch", "interactive", "local", "dry_run"}:
        raise ValueError("mode must be one of batch, interactive, local, dry_run")
    if mode == "batch" and not (campaign_dir / "submit.slurm").is_file():
        raise ValueError("batch mode requires submit.slurm")
    command = _launch_command(mode)
    plan = {
        "launch_plan_id": str(uuid.uuid4()),
        "created_at": _now(),
        "system": key,
        "campaign": campaign_dir.name,
        "campaign_dir": str(campaign_dir),
        "mode": mode,
        "workflow_script": "workflow.py",
        "batch_script": "submit.slurm" if mode == "batch" else None,
        "container": f"../containers/{key}/matensemble.sif",
        "command": command,
        "cwd": str(campaign_dir),
        "resources": _resources(account, nodes, walltime, queue, gpus, tasks),
        "requires_confirmation": True,
    }
    plan["command_hash"] = _plan_hash(plan)
    path = campaign_dir / "launch_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return ok(
        {"launch_plan": plan, "launch_plan_path": relative_to_workspace(path, cwd=cwd)},
        modified_files=[relative_to_workspace(path, cwd=cwd)] if path.exists() else [],
        commands_not_run=[command],
    )


def confirm_launch(
    launch_plan_id: str,
    *,
    timeout_seconds: int = 60,
    cwd: Path | None = None,
) -> dict[str, Any]:
    plan_path, plan = _find_plan("launch_plan.json", launch_plan_id, cwd=cwd)
    if _plan_hash(plan) != plan.get("command_hash"):
        return error("PLAN_MODIFIED", "launch plan command hash does not match stored plan")
    command = list(plan["command"])
    _validate_command(command)
    if shutil.which(command[0]) is None:
        return error("COMMAND_NOT_FOUND", f"required command not found on PATH: {command[0]}", commands_not_run=[command])
    completed = subprocess.run(
        command,
        cwd=str(Path(plan["cwd"])),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    stdout_path = Path(plan["cwd"]) / "launch_logs" / "sbatch_stdout.txt"
    stderr_path = Path(plan["cwd"]) / "launch_logs" / "sbatch_stderr.txt"
    stdout_path.parent.mkdir(exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    job_id = _parse_job_id(completed.stdout)
    return ok(
        {
            "launch_plan_id": launch_plan_id,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "job_id": job_id,
            "plan_path": relative_to_workspace(plan_path, cwd=cwd),
        },
        modified_files=[
            relative_to_workspace(stdout_path, cwd=cwd),
            relative_to_workspace(stderr_path, cwd=cwd),
        ],
        commands_run=[command],
    )


def prepare_container_pull_plan(
    system: str,
    *,
    force: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    workspace = scratch_workspace_root(cwd=cwd)
    target_dir = workspace / "containers" / key
    target = target_dir / "matensemble.sif"
    if target.exists() and not force:
        return ok({"system": key, "container": str(target), "already_exists": True})
    image = resolve_image_tag(key)
    if key == "linux":
        command = ["docker", "pull", image]
    elif key == "perlmutter":
        command = ["podman-hpc", "pull", image]
    else:
        command = ["apptainer", "pull", str(target), f"docker://{image}"]
    plan = _generic_plan("container_pull", key, command, cwd=str(target_dir), target=str(target))
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "pull_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return ok({"pull_plan": plan, "plan_path": relative_to_workspace(path, cwd=cwd)}, created_files=[relative_to_workspace(path, cwd=cwd)], commands_not_run=[command])


def confirm_container_pull(plan_id: str, *, timeout_seconds: int = 1800, cwd: Path | None = None) -> dict[str, Any]:
    return _confirm_generic_plan("pull_plan.json", plan_id, timeout_seconds=timeout_seconds, cwd=cwd)


def write_container_file(
    campaign: str,
    system: str,
    *,
    apt_packages: list[str] | None = None,
    python_packages: list[str] | None = None,
    backend: str = "apptainer",
    overwrite: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    profile = get_system_profile(key)
    if backend not in profile.container_backends:
        raise ValueError(f"backend must be one of {profile.container_backends}")
    apt = _validate_packages(apt_packages or [])
    py = _validate_packages(python_packages or [])
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    container_dir = campaign_dir / "container"
    container_dir.mkdir(exist_ok=True)
    filename = "Apptainer.def" if backend == "apptainer" else "Containerfile"
    target = container_dir / filename
    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(f"container file already exists: {target}")
    base = resolve_image_tag(key)
    text = _render_container_file(base, backend, apt, py)
    ensure_safe_generated_text(text)
    target.write_text(text, encoding="utf-8")
    rel = relative_to_workspace(target, cwd=cwd)
    return ok({"path": rel}, created_files=[] if existed else [rel], modified_files=[rel] if existed else [])


def prepare_container_build_plan(
    campaign: str,
    system: str,
    *,
    backend: str = "apptainer",
    cwd: Path | None = None,
) -> dict[str, Any]:
    key = normalize_system(system)
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    if backend == "apptainer":
        definition = campaign_dir / "container" / "Apptainer.def"
        target = campaign_dir / "container" / "matensemble-custom.sif"
        command = ["apptainer", "build", str(target), str(definition)]
    elif backend == "podman-hpc":
        definition = campaign_dir / "container" / "Containerfile"
        target = campaign_dir.name + "-matensemble-custom"
        command = ["podman-hpc", "build", "-t", target, "-f", str(definition), "."]
    elif backend == "docker" and key == "linux":
        definition = campaign_dir / "container" / "Containerfile"
        target = campaign_dir.name + "-matensemble-custom"
        command = ["docker", "build", "-t", target, "-f", str(definition), "."]
    else:
        raise ValueError("unsupported container build backend for system")
    if not definition.is_file():
        raise ValueError(f"container definition does not exist: {definition}")
    plan = _generic_plan("container_build", key, command, cwd=str(campaign_dir), target=str(target), campaign=campaign_dir.name)
    path = campaign_dir / "container" / "build_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return ok({"build_plan": plan, "plan_path": relative_to_workspace(path, cwd=cwd)}, created_files=[relative_to_workspace(path, cwd=cwd)], commands_not_run=[command])


def confirm_container_build(plan_id: str, *, timeout_seconds: int = 3600, cwd: Path | None = None) -> dict[str, Any]:
    return _confirm_generic_plan("build_plan.json", plan_id, timeout_seconds=timeout_seconds, cwd=cwd)


def get_job_status(job_id: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    if not str(job_id).isdigit():
        raise ValueError("job_id must be numeric")
    command = ["squeue", "-j", str(job_id), "-o", "%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R"]
    _validate_command(command)
    if shutil.which("squeue") is None:
        return error("COMMAND_NOT_FOUND", "required command not found on PATH: squeue", commands_not_run=[command])
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    return ok({"job_id": str(job_id), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}, commands_run=[command])


def prepare_cancel_job(job_id: str, *, cwd: Path | None = None) -> dict[str, Any]:
    if not str(job_id).isdigit():
        raise ValueError("job_id must be numeric")
    command = ["scancel", str(job_id)]
    plan = _generic_plan("cancel_job", "scheduler", command, cwd=str(scratch_workspace_root(cwd=cwd)), job_id=str(job_id))
    path = scratch_workspace_root(cwd=cwd) / f"cancel_{job_id}.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return ok({"cancel_plan": plan, "plan_path": relative_to_workspace(path, cwd=cwd)}, created_files=[relative_to_workspace(path, cwd=cwd)], commands_not_run=[command])


def confirm_cancel_job(plan_id: str, *, timeout_seconds: int = 30, cwd: Path | None = None) -> dict[str, Any]:
    return _confirm_generic_plan(f"cancel_*.json", plan_id, timeout_seconds=timeout_seconds, cwd=cwd)


def delete_campaign_plan(campaign: str, *, cwd: Path | None = None) -> dict[str, Any]:
    campaign_dir = _campaign_dir(campaign, cwd=cwd)
    plan = {
        "plan_id": str(uuid.uuid4()),
        "kind": "delete_campaign",
        "campaign": campaign_dir.name,
        "campaign_dir": str(campaign_dir),
        "created_at": _now(),
        "requires_confirmation": True,
    }
    plan["command_hash"] = _plan_hash(plan)
    path = campaign_dir / "delete_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return ok({"delete_plan": plan, "plan_path": relative_to_workspace(path, cwd=cwd)}, created_files=[relative_to_workspace(path, cwd=cwd)])


def confirm_delete_campaign(plan_id: str, *, cwd: Path | None = None) -> dict[str, Any]:
    plan_path, plan = _find_plan("delete_plan.json", plan_id, cwd=cwd)
    if _plan_hash(plan) != plan.get("command_hash"):
        return error("PLAN_MODIFIED", "delete plan hash does not match stored plan")
    campaign_dir = resolve_within_workspace(plan["campaign_dir"], cwd=cwd)
    deleted = relative_to_workspace(campaign_dir, cwd=cwd)
    shutil.rmtree(campaign_dir)
    return ok({"plan_path": str(plan_path), "deleted_campaign": deleted}, deleted_files=[deleted])


def create_matensemble_workflow_from_prompt(
    prompt: str,
    system: str,
    campaign_name: str,
    *,
    account: str | None = None,
    workflow_kind: str = "python_dag",
    cwd: Path | None = None,
) -> dict[str, Any]:
    context = get_system_context(system)
    if not context["ok"]:
        return context
    campaign_result = create_campaign(campaign_name, system, auto_suffix=True, cwd=cwd)
    if not campaign_result["ok"]:
        return campaign_result
    campaign = campaign_result["result"]["campaign"]
    created = list(campaign_result["created_files"])
    workflow = write_workflow(campaign, prompt, workflow_kind=workflow_kind, cwd=cwd)
    if not workflow["ok"]:
        return workflow
    created.extend(workflow["created_files"])
    warnings = []
    batch_result = None
    if account:
        batch_result = write_batch_script(campaign, system, account=account, cwd=cwd)
        if batch_result["ok"]:
            created.extend(batch_result["created_files"])
        else:
            warnings.append(batch_result["message"])
    else:
        warnings.append("batch script was not written because account is required")
    validation = validate_campaign(campaign, cwd=cwd)
    return ok(
        {
            "campaign": campaign,
            "campaign_dir": campaign_result["result"]["campaign_dir"],
            "system_context": context["result"],
            "workflow": workflow["result"],
            "batch": batch_result["result"] if batch_result and batch_result["ok"] else None,
            "validation": validation,
        },
        created_files=created,
        warnings=warnings,
    )


def _campaign_dir(campaign: str, *, cwd: Path | None = None) -> Path:
    name = validate_campaign_name(campaign)
    path = resolve_within_workspace(name, cwd=cwd)
    if not path.is_dir():
        raise ValueError(f"campaign does not exist: {name}")
    return path


def _validate_workflow_text(text: str) -> None:
    ensure_safe_generated_text(text)
    required = [
        "from matensemble.pipeline import Pipeline",
        "Pipeline(",
        "pipe.submit",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"workflow.py is missing required MatEnsemble patterns: {missing}")
    if "/home/" in text or "../.." in text:
        raise ValueError("workflow.py contains unsafe absolute or traversal paths")
    compile(text, "workflow.py", "exec")


def _render_batch(**kwargs: Any) -> str:
    key = kwargs["system"]
    account = kwargs["account"]
    queue = kwargs["queue"]
    nodes = int(kwargs["nodes"])
    walltime = kwargs["walltime"]
    gpus = int(kwargs["gpus"])
    tasks = int(kwargs["tasks"])
    backend = kwargs["container_backend"]
    container = kwargs["container_path"]
    campaign = kwargs["campaign"]
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by MatEnsemble MCP.",
        f"# System: {key}",
        f"# Campaign: {campaign}",
        f"# Source template: example_workflows/{key}/<selected>/run_batch.slurm",
        f"#SBATCH -A {account}",
        f"#SBATCH -J {slugify(campaign)[:32]}",
        "#SBATCH -o launch_logs/%x-%j.out",
        "#SBATCH -e launch_logs/%x-%j.err",
        f"#SBATCH -t {walltime}",
        f"#SBATCH -N {nodes}",
    ]
    if key == "perlmutter":
        lines.extend([f"#SBATCH --qos {queue}", "#SBATCH -C gpu", f"#SBATCH --ntasks-per-node={tasks}", f"#SBATCH --gpus-per-node={gpus}"])
    elif key in {"frontier", "pathfinder"}:
        lines.append(f"#SBATCH -p {queue}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            'CAMPAIGN_DIR="$(cd "$(dirname "$0")" && pwd)"',
            'cd "$CAMPAIGN_DIR"',
            "mkdir -p launch_logs",
            "",
        ]
    )
    if backend == "apptainer":
        lines.append(f"apptainer exec {container} flux start python workflow.py")
    elif backend == "podman-hpc":
        lines.append(f"podman-hpc run --rm --network=host --ipc=host -v \"$PWD:$PWD\" -w \"$PWD\" {container} flux start python workflow.py")
    else:
        lines.append(f"docker run --rm -v \"$PWD:$PWD\" -w \"$PWD\" {container} flux start --test-size={max(tasks, 1)} python workflow.py")
    return "\n".join(lines) + "\n"


def _resources(account: str | None, nodes: int, walltime: str, queue: str, gpus: int, tasks: int) -> dict[str, Any]:
    return {
        "account": account,
        "nodes": nodes,
        "walltime": walltime,
        "queue": queue,
        "gpus": gpus,
        "tasks": tasks,
    }


def _launch_command(mode: str) -> list[str]:
    if mode == "batch":
        return ["sbatch", "submit.slurm"]
    if mode == "interactive":
        return ["bash", "run_interactive.sh"]
    if mode == "local":
        return ["flux", "start", "--test-size=4", "python", "workflow.py"]
    return ["python", "-m", "py_compile", "workflow.py"]


def _validate_command(command: list[str]) -> None:
    if not command:
        raise ValueError("command is required")
    if command[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"command is not allowlisted: {command[0]}")
    if any(part in {"sudo"} for part in command):
        raise ValueError("sudo is not allowed")


def _plan_hash(plan: dict[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "command_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _find_plan(filename: str, plan_id: str, *, cwd: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = scratch_workspace_root(cwd=cwd)
    patterns = [filename] if "*" in filename else [f"**/{filename}"]
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            plan = json.loads(path.read_text(encoding="utf-8"))
            if plan.get("launch_plan_id") == plan_id or plan.get("plan_id") == plan_id:
                return path, plan
    raise ValueError(f"unknown launch_plan_id or plan_id: {plan_id}")


def _generic_plan(kind: str, system: str, command: list[str], *, cwd: str, **extra: Any) -> dict[str, Any]:
    _validate_command(command)
    plan = {
        "plan_id": str(uuid.uuid4()),
        "kind": kind,
        "created_at": _now(),
        "system": system,
        "command": command,
        "cwd": cwd,
        "requires_confirmation": True,
        **extra,
    }
    plan["command_hash"] = _plan_hash(plan)
    return plan


def _confirm_generic_plan(filename: str, plan_id: str, *, timeout_seconds: int, cwd: Path | None = None) -> dict[str, Any]:
    plan_path, plan = _find_plan(filename, plan_id, cwd=cwd)
    if _plan_hash(plan) != plan.get("command_hash"):
        return error("PLAN_MODIFIED", "plan hash does not match stored plan")
    command = list(plan["command"])
    _validate_command(command)
    if shutil.which(command[0]) is None:
        return error("COMMAND_NOT_FOUND", f"required command not found on PATH: {command[0]}", commands_not_run=[command])
    completed = subprocess.run(command, cwd=plan["cwd"], check=False, capture_output=True, text=True, timeout=timeout_seconds)
    return ok(
        {
            "plan_id": plan_id,
            "plan_path": relative_to_workspace(plan_path, cwd=cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        commands_run=[command],
    )


def _parse_job_id(stdout: str) -> str | None:
    match = JOB_ID_RE.search(stdout)
    return match.group(1) if match else None


def _validate_packages(packages: list[str]) -> list[str]:
    bad = [pkg for pkg in packages if not PACKAGE_RE.match(pkg)]
    if bad:
        raise ValueError(f"unsupported package names: {bad}")
    return packages


def _render_container_file(base_image: str, backend: str, apt_packages: list[str], python_packages: list[str]) -> str:
    apt_line = " ".join(apt_packages)
    py_line = " ".join(python_packages)
    if backend == "apptainer":
        lines = [
            f"Bootstrap: docker",
            f"From: {base_image}",
            "",
            "%post",
            "    # ---- MCP GENERATED USER PACKAGES START ----",
        ]
        if apt_packages:
            lines.extend(["    apt-get update", f"    apt-get install -y {apt_line}"])
        if python_packages:
            lines.append(f"    uv pip install --system {py_line}")
        lines.append("    # ---- MCP GENERATED USER PACKAGES END ----")
        return "\n".join(lines) + "\n"
    lines = [
        f"FROM {base_image}",
        "# ---- MCP GENERATED USER PACKAGES START ----",
    ]
    if apt_packages:
        lines.append(f"RUN apt-get update && apt-get install -y {apt_line} && rm -rf /var/lib/apt/lists/*")
    if python_packages:
        lines.append(f"RUN uv pip install --system {py_line}")
    lines.append("# ---- MCP GENERATED USER PACKAGES END ----")
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
