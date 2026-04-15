# zulipmcp

**Public open-source repo.** All code, commits, and PRs are visible to the world.

## Rules

- Nothing WindBorne-specific. No internal URLs, emails, Zulip instances, or infra references.
- No internal context in PRs, commits, or comments.
- Elegant, general code. Type hints everywhere. Minimal dependencies.

## Architecture

Two files carry all the weight:

- **`core.py`** — Pure functions + Zulip API wrappers. No MCP dependency. Returns Python objects. This is also the library's public API (re-exported via `__init__.py`).
- **`mcp.py`** — MCP tool layer + `SessionState`. Thin wrappers that call `core.py` and format results as strings. Hooks system lives here.

Separation is load-bearing: `core.py` must stay MCP-agnostic so it works as a standalone library import.

## Gotchas

- **`listen()` is async, everything else is sync.** `listen()` runs `get_events()` in a thread executor to interleave MCP keepalive pings. Don't make other tools async — FastMCP handles sync tools fine.
- **Event queue narrow does NOT filter reactions.** `is_dismiss_reaction()` receives reactions from all streams. It fetches the reacted-on message to verify stream/topic match. Don't remove that check.
- **`_session` is module-level singleton.** One session per MCP server process. `set_context()` re-initializes it. Don't add multi-session support — MCP is one-client-per-server.
- **`reply()` checks for missed messages before sending.** This catches messages that arrived while the LLM was thinking. `last_seen_message_id` bookkeeping is subtle — trace it carefully before changing.
- **Two dismiss-check paths exist intentionally.** `listen()` catches dismiss reactions via events. `reply()` catches them via REST API poll. Both are needed — the user might react while the bot is in tool execution (not listening).
- **Private stream security is default-deny.** Unset `BOT_ALLOWED_PRIVATE_STREAMS` = no private stream access. `BOT_ALLOWED_WRITE_STREAMS` is the opposite (unset = all writes allowed) for backwards compatibility. These asymmetries are intentional.
- **`configure()` must be called before `run_server()`.** Hooks are registered on module-level `_hooks` dict. `run_server()` may auto-init a session that reads hook state.
- **Zulip's `remove_reaction` needs `reaction_type`** for custom emoji. The `core.py` wrapper hardcodes `"realm_emoji"`. Standard Unicode emoji don't need it but the param doesn't hurt.
- **Message combining merges consecutive same-user messages within 2 min.** The combined message keeps the *last* message's ID. This affects `last_seen_message_id` tracking in callers.
- **`/nobots` and `/nb` filtering** is applied in `filter_for_bot()` which is called by `format_messages()` and explicitly in `listen()`. New message-fetching paths must also filter.
