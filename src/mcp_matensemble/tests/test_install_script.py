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
