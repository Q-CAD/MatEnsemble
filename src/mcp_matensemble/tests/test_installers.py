from __future__ import annotations

import json
from pathlib import Path
import subprocess

from mcp_matensemble.installers.cli import main


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_agent_install_writes_workspace_and_wrappers(tmp_path):
    scratch = tmp_path / "scratch"
    install_dir = tmp_path / "bin"
    workspace = scratch / "matensemble_campaigns"
    config_dir = tmp_path / "config"

    import os

    old_scratch = os.environ.get("SCRATCH")
    os.environ["SCRATCH"] = str(scratch)
    try:
        main(
            [
                "--system",
                "frontier",
                "--workspace",
                str(workspace),
                "--install-dir",
                str(install_dir),
                "--config-dir",
                str(config_dir),
            ]
        )
    finally:
        if old_scratch is None:
            os.environ.pop("SCRATCH", None)
        else:
            os.environ["SCRATCH"] = old_scratch

    mcp_config = json.loads((workspace / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert mcp_config["servers"]["matensemble"]["command"] == str(
        install_dir / "mcp-matensemble-frontier"
    )
    wrapper = (install_dir / "mcp-matensemble-frontier").read_text(encoding="utf-8")
    assert "uv" in wrapper
    assert "run mcp-matensemble" in wrapper
    assert "exec" in wrapper
    site_cli = (install_dir / "matensemble").read_text(encoding="utf-8")
    stable_cli = (REPO_ROOT / "src" / "cli" / "matensemble-frontier").read_text(
        encoding="utf-8"
    )
    assert site_cli == stable_cli
    assert (workspace / "README.md").exists()
    assert (workspace / ".matensemble-mcp.toml").exists()


def test_stable_perlmutter_cli_matches_flux_gpu_launch_requirements():
    script = (REPO_ROOT / "src" / "cli" / "matensemble-perlmutter").read_text(
        encoding="utf-8"
    )
    checked = subprocess.run(
        ["bash", "-n"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )

    assert checked.returncode == 0, checked.stderr
    assert "expand_nodelist" in script
    assert '"nodelist": nodes' in script
    assert '"nodelist": [nodes]' not in script
    assert "LD_PRELOAD=/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" in script
    assert "-e \"LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${ld_path}\"" in script
    assert "-v /dev/shm:/dev/shm" in script
    assert "-v /dev/cxi0:/dev/cxi0" in script
    assert "-v /dev/xpmem:/dev/xpmem" in script
    assert "--device /dev/nvidia0" in script
    assert "--device /dev/nvidia-uvm" in script
    assert "FLUX_CONF_DIR=/tmp/fluxcfg flux start" in script
