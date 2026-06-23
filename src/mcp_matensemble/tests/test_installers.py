from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tomllib

from mcp_matensemble.installers.cli import main


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_agent_install_writes_workspace_and_wrappers(tmp_path):
    scratch = tmp_path / "scratch"
    install_dir = tmp_path / "bin"
    workspace = scratch / "matensemble_campaigns"
    config_dir = tmp_path / "config"
    codex_config = tmp_path / "codex" / "config.toml"

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
                "--codex-config",
                str(codex_config),
            ]
        )
    finally:
        if old_scratch is None:
            os.environ.pop("SCRATCH", None)
        else:
            os.environ["SCRATCH"] = old_scratch

    mcp_config = json.loads(
        (workspace / ".vscode" / "mcp.json").read_text(encoding="utf-8")
    )
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
    codex = tomllib.loads(codex_config.read_text(encoding="utf-8"))
    codex_server = codex["mcp_servers"]["matensemble"]
    assert codex_server["command"] == str(install_dir / "mcp-matensemble-frontier")
    assert codex_server["cwd"] == str(workspace)
    assert codex_server["startup_timeout_sec"] == 120
    assert codex_server["env"]["SCRATCH"] == str(scratch)


def test_agent_install_updates_existing_codex_config(tmp_path):
    scratch = tmp_path / "scratch"
    install_dir = tmp_path / "bin"
    workspace = scratch / "matensemble_campaigns"
    config_dir = tmp_path / "config"
    codex_config = tmp_path / "codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text(
        'model = "gpt-5.5"\n\n'
        "[mcp_servers.node_repl]\n"
        'command = "/bin/node_repl"\n\n'
        "[mcp_servers.matensemble]\n"
        'command = "/old/matensemble"\n'
        'cwd = "/old/workspace"\n\n'
        "[mcp_servers.matensemble.env]\n"
        'SCRATCH = "/old/scratch"\n\n'
        "[projects.\"/tmp/project\"]\n"
        'trust_level = "trusted"\n',
        encoding="utf-8",
    )

    import os

    old_scratch = os.environ.get("SCRATCH")
    os.environ["SCRATCH"] = str(scratch)
    try:
        main(
            [
                "--system",
                "linux",
                "--workspace",
                str(workspace),
                "--install-dir",
                str(install_dir),
                "--config-dir",
                str(config_dir),
                "--codex-config",
                str(codex_config),
            ]
        )
    finally:
        if old_scratch is None:
            os.environ.pop("SCRATCH", None)
        else:
            os.environ["SCRATCH"] = old_scratch

    codex = tomllib.loads(codex_config.read_text(encoding="utf-8"))

    assert codex["model"] == "gpt-5.5"
    assert codex["mcp_servers"]["node_repl"]["command"] == "/bin/node_repl"
    assert codex["mcp_servers"]["matensemble"]["command"] == str(
        install_dir / "mcp-matensemble-linux"
    )
    assert codex["mcp_servers"]["matensemble"]["env"]["SCRATCH"] == str(scratch)
    assert codex["projects"]["/tmp/project"]["trust_level"] == "trusted"


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
