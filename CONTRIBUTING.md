# Contributing to zulipmcp

## Development Setup

```bash
git clone https://github.com/windborne/zulipmcp.git
cd zulipmcp
uv sync
```

## Running Locally

Place a `.zuliprc` in the project root (see [Zulip bot setup](https://zulip.com/help/add-a-bot-or-integration)):

```bash
# MCP server (stdio)
uv run python -m zulipmcp.mcp

# Listener (watches for @mentions)
uv run python -m zulipmcp.listener
```

## Code Style

- Type hints on all public functions
- Keep `core.py` MCP-agnostic — it's also the library API
- Keep `mcp.py` as a thin wrapper over `core.py`
- Minimal dependencies — don't add packages for things the stdlib can do

## Pull Requests

- Keep PRs focused — one concern per PR
- Describe *what* changed and *why* in the PR description
- Don't include internal links, emails, or references to private systems
