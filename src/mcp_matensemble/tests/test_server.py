from __future__ import annotations

from mcp_matensemble import server


def test_server_parser_requires_supported_system():
    args = server.build_parser().parse_args(["--system", "frontier"])

    assert args.system == "frontier"
