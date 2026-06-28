from __future__ import annotations

import asyncio

from mcp_matensemble import server


def test_server_parser_requires_supported_system():
    args = server.build_parser().parse_args(["--system", "frontier"])

    assert args.system == "frontier"


def test_server_registers_dashboard_prompt_and_batch_tool():
    mcp = server.create_server("frontier")

    tools = asyncio.run(mcp.list_tools())
    prompts = asyncio.run(mcp.list_prompts())

    assert "get_example_batch_scripts" in {tool.name for tool in tools}
    assert "start_dashboard" in {prompt.name for prompt in prompts}
