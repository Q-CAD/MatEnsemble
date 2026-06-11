from __future__ import annotations

import json

from mcp_matensemble.installers.cli import main


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
    assert "perlmutter" not in site_cli
    assert "frontier" in site_cli
    assert (workspace / "README.md").exists()
    assert (workspace / ".matensemble-mcp.toml").exists()
