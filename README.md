# zulipmcp

Clean Zulip interface for LLMs. Fetches, caches, and formats Zulip messages into concise XML suitable for LLM consumption. Usable as a Python library or as an MCP server.

## Requirements

- Python >=3.10, managed with [uv](https://docs.astral.sh/uv/)
- A `.zuliprc` file in the project root for Zulip API auth
- SSH access to the [windborne](https://github.com/windborne) GitHub org (private repo)

## Using with Claude Code

Install the package:

```bash
uv add zulipmcp --git ssh://git@github.com/windborne/zulipmcp.git
```

Add a `.zuliprc` file to your project root with your Zulip bot credentials.

Add the MCP server to your `.mcp.json`:

```json
{
  "mcpServers": {
    "zulip": {
      "command": "uv",
      "args": ["run", "python", "-m", "zulipmcp.mcp"]
    }
  }
}
```

Restart Claude Code. The Zulip tools should now be available.

## Entry Points

| Entry Point | Description |
|---|---|
| `uv run python -m zulipmcp.mcp` | MCP server for Claude Code / MCP clients |
| `uv run python -m zulipmcp.mcp --transport sse` | MCP server over SSE (for remote/web clients) |

## Style Notes

Keep code in core.py elegant, short, and simple.
