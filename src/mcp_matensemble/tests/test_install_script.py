from __future__ import annotations

import subprocess
from pathlib import Path


def test_root_install_script_has_valid_bash_syntax():
    root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        ["bash", "-n", str(root / "install.sh")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_container_engine_detection_priority(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    for name in ("podman-hpc", "podman", "docker", "apptainer"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; PATH="$2" detect_container_engine',
            "bash",
            str(root / "install.sh"),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "apptainer"


def test_install_root_is_nested_under_matensemble(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    parent = tmp_path / "software"

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; response="$2"; prompt_read() { printf "%s\\n" "$response"; }; choose_install_root',
            "bash",
            str(root / "install.sh"),
            str(parent),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(parent / "MatEnsemble")


def test_explicit_matensemble_install_root_is_not_duplicated(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    install_root = tmp_path / "MatEnsemble"

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; response="$2"; prompt_read() { printf "%s\\n" "$response"; }; choose_install_root',
            "bash",
            str(root / "install.sh"),
            str(install_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(install_root)
