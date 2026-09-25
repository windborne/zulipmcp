"""Tests for the auto_typing_start hook.

Contract: with the default (True) a typing "start" is sent on set_context()
and after reply(); with configure(auto_typing_start=False) set_context()
sends nothing and reply() sends a "stop" instead, while the "stop" on
listen()/end_session() is always sent and the explicit typing() tool is
unaffected.
"""
import asyncio
import importlib

import pytest

import zulipmcp.core as zulip_core

# The package re-exports its FastMCP instance as `zulipmcp.mcp`, which
# shadows the submodule for `import zulipmcp.mcp as ...` — load it by name.
mcp = importlib.import_module("zulipmcp.mcp")


def _fn(tool):
    """Return the plain function whether or not the decorator wrapped it (fastmcp has done both)."""
    return getattr(tool, "fn", tool)


@pytest.fixture
def typing_ops(monkeypatch):
    """Record every send_typing op; stub the Zulip calls the tools make."""
    ops: list[str] = []
    monkeypatch.setattr(zulip_core, "send_typing", lambda s, t, op="start": ops.append(op) or {"result": "success"})
    monkeypatch.setattr(zulip_core, "send_message", lambda s, t, c: {"result": "success", "id": 42})
    monkeypatch.delenv("BOT_ALLOWED_WRITE_STREAMS", raising=False)
    monkeypatch.setattr(zulip_core, "is_stream_private", lambda stream: False)
    monkeypatch.setattr(zulip_core, "add_reaction", lambda *a, **k: {"result": "success"})
    monkeypatch.setattr(zulip_core, "check_dismissed", lambda *a, **k: None)
    monkeypatch.setattr(zulip_core, "fetch_new_messages", lambda *a, **k: [])
    monkeypatch.setattr(mcp, "_init_session", lambda stream, topic, num_messages=0: "Session context set")
    monkeypatch.setattr(mcp, "_write_exit_markers", lambda: None)
    monkeypatch.setitem(mcp._hooks, "auto_typing_start", True)
    mcp._session.reset()
    mcp._session.stream, mcp._session.topic, mcp._session.active = "eng", "topic", True
    monkeypatch.setattr(mcp._session, "my_user_id", 7)  # reset() keeps it; restore on teardown
    yield ops
    mcp._session.reset()


def test_default_starts_on_set_context(typing_ops):
    _fn(mcp.set_context)("eng", "topic")
    assert typing_ops == ["start"]


def test_default_restarts_after_reply(typing_ops):
    assert _fn(mcp.reply)("hi").startswith("Message sent")
    assert typing_ops == ["start"]


def test_disabled_suppresses_set_context_start(typing_ops):
    mcp.configure(auto_typing_start=False)
    _fn(mcp.set_context)("eng", "topic")
    assert typing_ops == []


def test_disabled_stops_after_reply(typing_ops):
    mcp.configure(auto_typing_start=False)
    assert _fn(mcp.reply)("hi").startswith("Message sent")
    assert typing_ops == ["stop"]


def test_disabled_reply_stop_is_after_send(typing_ops, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(zulip_core, "send_message", lambda s, t, c: order.append("send") or {"result": "success", "id": 42})
    monkeypatch.setattr(zulip_core, "send_typing", lambda s, t, op="start": order.append(op) or {"result": "success"})
    mcp.configure(auto_typing_start=False)
    _fn(mcp.reply)("hi")
    assert order == ["send", "stop"]


def test_disabled_reply_stop_failure_never_raises(typing_ops, monkeypatch):
    mcp.configure(auto_typing_start=False)
    monkeypatch.setattr(zulip_core, "send_typing", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _fn(mcp.reply)("hi").startswith("Message sent")


def test_disabled_still_stops_on_end_session(typing_ops):
    mcp.configure(auto_typing_start=False)
    _fn(mcp.end_session)("")
    assert typing_ops == ["stop"]


def test_disabled_still_stops_on_listen(typing_ops, monkeypatch):
    mcp.configure(auto_typing_start=False)
    # A pre-existing dismiss returns listen() right after its typing stop,
    # before any event queue is registered.
    mcp._session.last_sent_message_id = 1
    monkeypatch.setattr(zulip_core, "check_dismissed", lambda *a, **k: "stop_sign")
    result = asyncio.run(_fn(mcp.listen)(0.001, ctx=None))
    assert result.startswith("Session dismissed")
    assert typing_ops == ["stop"]


def test_disabled_leaves_explicit_typing_tool(typing_ops):
    mcp.configure(auto_typing_start=False)
    assert _fn(mcp.typing)() == "Typing indicator start."
    assert typing_ops == ["start"]


def test_start_failure_never_raises(typing_ops, monkeypatch):
    monkeypatch.setattr(zulip_core, "send_typing", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _fn(mcp.reply)("hi").startswith("Message sent")


def test_unknown_hook_still_rejected():
    with pytest.raises(ValueError):
        mcp.configure(auto_typing=False)
