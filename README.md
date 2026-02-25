# zulipmcp

<img width="1446" height="752" alt="image" src="https://github.com/user-attachments/assets/6e6bbed7-ed19-4c4a-a9f2-48468dc9a570" />


Clean Zulip interface for LLMs. Fetches, caches, and formats Zulip messages into concise XML suitable for LLM consumption. Usable as a Python library or as an MCP server.

## Requirements

- Python >=3.10, managed with [uv](https://docs.astral.sh/uv/)
- A `.zuliprc` file in the project root for Zulip API auth

## Using with Claude Code

1. Install the package:

   ```bash
   uv add zulipmcp --git https://github.com/windborne/zulipmcp.git
   ```

2. Add a `.zuliprc` file to your project root with your Zulip bot credentials. See [Configuring the Python bindings](https://zulip.com/api/configuring-python-bindings) for details on the file format and how to download one from your Zulip organization.

3. Add the MCP server to your `.mcp.json`:

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

4. Restart Claude Code. The Zulip tools should now be available.

## Entry Points

| Entry Point | Description |
|---|---|
| `uv run python -m zulipmcp.mcp` | MCP server for Claude Code / MCP clients |
| `uv run python -m zulipmcp.mcp --transport sse` | MCP server over SSE (for remote/web clients) |
| `uv run python -m zulipmcp.listener` | Listener: watches for @mentions, spawns Claude Code sessions |

## Listener

The optional `zulipmcp.listener` module watches Zulip for @mentions and spawns one headless Claude Code session per (stream, topic). It's the glue between Zulip events and Claude Code — the MCP server handles all the Zulip tools, the listener just handles lifecycle.

```bash
# Minimal — uses ./.zuliprc, ./.mcp.json (if present), and the bundled default prompt
uv run python -m zulipmcp.listener

# Full — override MCP config and system prompt
uv run python -m zulipmcp.listener \
    --mcp-config .mcp.json \
    --system-prompt agent.md \
    --log-dir ./logs
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--zuliprc` | `./.zuliprc` | Path to `.zuliprc` (resolved relative to current working directory) |
| `--mcp-config` | `./.mcp.json` | Path to `.mcp.json` for Claude Code sessions (used only if the file exists) |
| `--system-prompt` | `zulipmcp/default_system_prompt.md` | Appended system prompt file (default path is resolved relative to `listener.py`, not the current working directory) |
| `--working-dir` | `.` | Working directory for spawned sessions |
| `--claude-command` | `claude` | Claude CLI binary name or path |
| `--log-dir` | `./logs` | Directory for session log files |

Each session gets `TRIGGER_MESSAGE_ID` and `SESSION_USER_EMAIL` set automatically so `set_context()` anchors to the @mention and hooks can identify the requester.

The listener is deliberately minimal (~90 lines of code). It omits concurrency caps, workspace isolation, staleness watchdogs, and dashboards — add those when you need them.

## Key design details

### Listening for messages

The `listen` tool uses Zulip's [real-time events API](https://zulip.com/api/real-time-events) (long-polling) instead of repeated `GET /messages` calls. On entry it catches up on any messages since `last_seen_message_id`, subscribes the bot to the stream if needed, registers a narrowed event queue for the stream/topic, and then long-polls via `GET /events`. The server blocks until a message arrives or ~90 seconds elapse (heartbeat), making this ~30x more efficient than polling every 2 seconds. If the queue expires (`BAD_EVENT_QUEUE_ID`), it re-registers automatically. The queue is deleted in a `finally` block on exit.

A `robot_ear` emoji is added to the last message as a visual indicator while listening and removed when listening stops. MCP keepalive pings are sent via `ctx.info()` after each long-poll cycle.

### No missed messages on reply

When `reply` is called, it checks for new messages *before* sending. If anyone posted while the LLM was thinking, those messages are fetched and returned alongside the "message sent" confirmation. This way the LLM always sees what it missed and can react accordingly. The `last_seen_message_id` is updated to whichever is newest — the missed messages or the sent message — so nothing falls through the cracks.

## Environment Variables

| Variable | Description |
|---|---|
| `TRIGGER_MESSAGE_ID` | Message ID that triggered the session (e.g. the @mention). Sets the listen anchor so the agent doesn't miss messages after the trigger. |
| `SESSION_USER_EMAIL` | Email of the human who triggered the session. Stored on `SessionState` for hooks. |
| `ZULIPMCP_CACHE_DIR` | Override the disk cache directory (defaults to system temp dir). |

## Logging

Logs are written to `/tmp/zulipmcp_logs` by default. Override with the `ZULIPMCP_LOG_DIR` environment variable.

## Style Notes

Keep code in core.py elegant, short, and simple.
