# mcp-matensemble

Small MCP server for MatEnsemble agents.

The server intentionally exposes only MatEnsemble context, deterministic
container tag/build guidance, and dashboard lifecycle helpers. It does not create
campaigns, submit jobs, cancel jobs, execute container setup, or add a separate
safety/guardrail layer.

```bash
uv run --package mcp-matensemble mcp-matensemble --system frontier
```
