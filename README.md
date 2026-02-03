# zulipmcp

Clean Zulip interface for LLMs. Fetches, caches, and formats Zulip messages into concise XML suitable for LLM consumption. Usable as a Python library or as an MCP server.

## Environment

- Python >=3.10, managed with `uv`
- Requires a `.zuliprc` file in the project root for Zulip API auth

## Style Notes

Keep code in zulip_core.py elegant, short, and simple.

## Entry Points

| Entry Point | Description |
|---|---|
| `uv run python zulip_mcp.py` | MCP server — thin wrapper over `zulip_core` for Claude Code / MCP clients. |