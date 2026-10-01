"""Tests for DM sessions: set_dm_context(), and reply()/listen()/end_session() in a DM.

Contract: a DM session targets the conversation with exactly the given
participants (the bot's own address is dropped). reply() and end_session()
send a direct message; listen() receives every DM (event-queue narrows can't
filter by participants) but only returns messages from this conversation.
"""
import asyncio
import importlib

import pytest

import zulipmcp.core as zulip_core
from zulipmcp.agent_backends import bootstrap_prompt

# The package re-exports its FastMCP instance as `zulipmcp.mcp`, which
# shadows the submodule for `import zulipmcp.mcp as ...` — load it by name.
mcp = importlib.import_module("zulipmcp.mcp")

BOT_ID = 7
BOT_EMAIL = "bot@example.com"


def _fn(tool):
    """Return the plain function whether or not the decorator wrapped it (fastmcp has done both)."""
    return getattr(tool, "fn", tool)


def _dm(msg_id: int, sender_id: int, *peers: tuple[int, str], content: str = "hi") -> dict:
    """A DM message between the bot and the given (user_id, email) peers."""
    recipients = [{"id": BOT_ID, "email": BOT_EMAIL}] + [{"id": i, "email": e} for i, e in peers]
    return {
        "id": msg_id, "type": "private", "sender_id": sender_id,
        "sender_full_name": f"user{sender_id}", "content": content,
        "timestamp": 1_700_000_000 + msg_id, "display_recipient": recipients,
    }


ALICE = (11, "Alice@Example.com")
BOB = (12, "bob@example.com")


@pytest.fixture
def calls(monkeypatch):
    """Stub the Zulip calls DM sessions make; record sends and typing."""
    log: dict[str, list] = {"dm_sends": [], "stream_sends": [], "typing": []}
    monkeypatch.setattr(zulip_core, "get_profile",
                        lambda: {"result": "success", "user_id": BOT_ID, "email": BOT_EMAIL})
    monkeypatch.setattr(zulip_core, "get_dm_messages", lambda emails, num_messages=20: [])
    monkeypatch.setattr(zulip_core, "get_emoji_info", lambda: (0, set()))
    monkeypatch.setattr(zulip_core, "send_direct_message",
                        lambda to, c: log["dm_sends"].append((to, c)) or {"result": "success", "id": 100})
    monkeypatch.setattr(zulip_core, "send_message",
                        lambda s, t, c: log["stream_sends"].append((s, t, c)) or {"result": "success", "id": 101})
    monkeypatch.setattr(zulip_core, "send_dm_typing",
                        lambda emails, op="start": log["typing"].append((tuple(emails), op)) or {"result": "success"})
    monkeypatch.setattr(zulip_core, "check_dismissed", lambda *a, **k: None)
    monkeypatch.setattr(zulip_core, "fetch_new_dm_messages", lambda *a, **k: [])
    monkeypatch.setattr(mcp, "_write_exit_markers", lambda: None)
    monkeypatch.delenv("TRIGGER_MESSAGE_ID", raising=False)
    mcp._session.reset()
    monkeypatch.setattr(mcp._session, "my_user_id", None)  # reset() keeps it; restore on teardown
    yield log
    mcp._session.reset()


def test_dm_peers_excludes_bot_and_lowercases():
    assert zulip_core.dm_peers(_dm(1, 11, ALICE, BOB), BOT_ID) == frozenset(
        {"alice@example.com", "bob@example.com"})


def test_dm_peers_is_none_for_stream_messages():
    assert zulip_core.dm_peers({"display_recipient": "general"}, BOT_ID) is None


def test_set_dm_context_normalizes_participants(calls):
    result = _fn(mcp.set_dm_context)(["Bob@example.com", "alice@example.com", BOT_EMAIL])
    assert result.startswith("Session context set to DM with alice@example.com, bob@example.com")
    assert mcp._session.dm == ["alice@example.com", "bob@example.com"]
    assert mcp._session.stream is None
    assert calls["typing"] == [(("alice@example.com", "bob@example.com"), "start")]


def test_set_dm_context_rejects_only_the_bot(calls):
    assert _fn(mcp.set_dm_context)([BOT_EMAIL]).startswith("Error:")
    assert not mcp._session.active


def test_reply_and_end_session_send_dms(calls):
    _fn(mcp.set_dm_context)(["alice@example.com"])
    assert _fn(mcp.reply)("hello").startswith("Message sent (id: 100)")
    ended = _fn(mcp.end_session)("bye")
    assert ended == "Session ended. Was chatting in DM with alice@example.com"
    assert [c for _, c in calls["dm_sends"]] == ["hello", "bye | 0s"]
    assert all(to == ["alice@example.com"] for to, _ in calls["dm_sends"])
    assert calls["stream_sends"] == []


def test_reply_skips_stream_write_allowlist(calls, monkeypatch):
    monkeypatch.setenv("BOT_ALLOWED_WRITE_STREAMS", "only-this-stream")
    _fn(mcp.set_dm_context)(["alice@example.com"])
    assert _fn(mcp.reply)("hello").startswith("Message sent")


class _Ctx:
    async def info(self, *a, **k) -> None:
        pass

    async def report_progress(self, *a, **k) -> None:
        pass


def test_listen_returns_only_this_conversation(calls, monkeypatch):
    other_dm = _dm(201, 12, BOB, content="not for this session")
    group_dm = _dm(202, 11, ALICE, BOB, content="group chat, also not this session")
    own = _dm(203, BOT_ID, ALICE, content="bot's own message")
    mine = _dm(204, 11, ALICE, content="for this session")
    batches = [[other_dm, group_dm], [own, mine]]

    monkeypatch.setattr(zulip_core, "register_dm_event_queue", lambda: ("q", -1, 90))
    monkeypatch.setattr(zulip_core, "register_event_queue",
                        lambda *a: pytest.fail("DM session must not register a stream queue"))
    monkeypatch.setattr(zulip_core, "ensure_subscribed",
                        lambda *a: pytest.fail("DM session must not subscribe to a stream"))
    monkeypatch.setattr(zulip_core, "get_events", lambda q, last, timeout: (batches.pop(0), [], last + 1))
    monkeypatch.setattr(zulip_core, "delete_event_queue", lambda q: None)
    monkeypatch.setattr(zulip_core, "check_reactions_on", lambda msg_id: "")
    monkeypatch.setattr(zulip_core, "add_reaction", lambda *a, **k: {"result": "success"})
    monkeypatch.setattr(zulip_core, "remove_reaction", lambda *a, **k: {"result": "success"})

    _fn(mcp.set_dm_context)(["alice@example.com"])
    out = asyncio.run(_fn(mcp.listen)(1, _Ctx()))

    assert "for this session" in out
    assert "not for this session" not in out
    assert "group chat" not in out
    assert "bot's own message" not in out
    assert mcp._session.last_seen_message_id == 204


def test_dismiss_reaction_checks_dm_participants(monkeypatch):
    msg = _dm(300, BOT_ID, ALICE)
    client = type("C", (), {"call_endpoint": lambda self, **k: {"result": "success", "message": msg}})()
    monkeypatch.setattr(zulip_core, "get_client", lambda: client)
    event = {"op": "add", "emoji_name": "stop_sign", "user_id": 11, "message_id": 300}
    assert zulip_core.is_dismiss_reaction(event, BOT_ID, dm=frozenset({"alice@example.com"}))
    assert not zulip_core.is_dismiss_reaction(event, BOT_ID, dm=frozenset({"bob@example.com"}))


def test_bootstrap_prompt_for_dm():
    prompt = bootstrap_prompt("", "", dm=["alice@example.com"])
    assert prompt.startswith("Call set_dm_context(['alice@example.com']) to begin")
    assert bootstrap_prompt("eng", "t").startswith("Call set_context('eng', 't') to begin")
