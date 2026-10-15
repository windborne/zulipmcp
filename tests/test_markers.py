"""unread/mentioned markers on formatted messages."""
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pytest

from zulipmcp import core

_MISSING = object()


def msg(msg_id: int, flags: Any = _MISSING, timestamp: int = 1000, sender_id: int = 5) -> dict:
    m: dict = {
        "id": msg_id, "timestamp": timestamp, "type": "stream",
        "display_recipient": "general", "subject": "t", "content": "hi",
        "sender_full_name": "Alice", "sender_id": sender_id,
    }
    if flags is not _MISSING:
        m["flags"] = flags
    return m


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], list[int]]:
    """Install a client whose get_profile returns (or raises) ``result``; returns the call log."""
    monkeypatch.setattr(core, "TIMEZONE", ZoneInfo("UTC"))

    def install(result: Any) -> list[int]:
        calls: list[int] = []

        class FakeClient:
            def get_profile(self) -> dict:
                calls.append(1)
                if isinstance(result, BaseException):
                    raise result
                return result

        fake = FakeClient()
        monkeypatch.setattr(core, "get_client", lambda: fake)
        return calls

    return install


HUMAN = {"result": "success", "is_bot": False, "user_id": 1}
BOT = {"result": "success", "is_bot": True, "user_id": 2}


def tag(out: str) -> str:
    return out.splitlines()[0]


def test_human_read(profile: Callable) -> None:
    profile(HUMAN)

    assert "unread=" not in core.format_messages([msg(1, ["read"])])


def test_missing_flags_has_no_markers_and_skips_profile(profile: Callable) -> None:
    calls = profile(HUMAN)
    out = core.format_messages([msg(1)])

    assert tag(out) == '<msg user="Alice" time="1970-01-01 00:16:40 UTC" id="1">'
    assert calls == []


def test_bot_never_unread_but_mentioned(profile: Callable) -> None:
    profile(BOT)
    out = core.format_messages([msg(1, ["mentioned"])])

    assert "unread=" not in out
    assert 'mentioned="true"' in out


@pytest.mark.parametrize("flag", [
    "mentioned", "stream_wildcard_mentioned", "topic_wildcard_mentioned",
    "group_mentioned", "wildcard_mentioned",
])
def test_every_mention_flag_marks_mentioned(profile: Callable, flag: str) -> None:
    profile(BOT)

    assert 'mentioned="true"' in core.format_messages([msg(1, ["read", flag])])


def test_other_flags_do_not_mark_mentioned(profile: Callable) -> None:
    profile(BOT)

    assert "mentioned=" not in core.format_messages([msg(1, ["read", "starred", "historical"])])


def test_marker_order(profile: Callable) -> None:
    profile(HUMAN)
    m = msg(7, ["mentioned"])
    m["reactions"] = [{"emoji_name": "tada"}]

    assert tag(core.format_messages([m])).endswith(
        'id="7" unread="true" mentioned="true" reactions=":tada:">'
    )


def test_own_messages_never_unread(profile: Callable) -> None:
    profile(HUMAN)
    out = core.format_messages([msg(1, [], sender_id=HUMAN["user_id"]), msg(2, [], timestamp=5000)])
    own, other = (line for line in out.splitlines() if line.startswith("<msg "))

    assert "unread=" not in own
    assert 'unread="true"' in other


@pytest.mark.parametrize("result, retried", [
    (ConnectionError("network down"), True),
    ({"result": "error", "msg": "unauthorized"}, True),
    ({"result": "success"}, False),
])
def test_profile_failure_means_no_unread(profile: Callable, result: Any, retried: bool) -> None:
    calls = profile(result)
    out = core.format_messages([msg(1, ["mentioned"]), msg(2, [], timestamp=5000)])

    assert "unread=" not in out
    assert out.count('mentioned="true"') == 1
    assert calls == [1]

    # A failed lookup isn't cached, so the next call retries; a definitive answer is.
    core.format_messages([msg(3, [])])
    assert calls == ([1, 1] if retried else [1])


@pytest.mark.parametrize("first, second, unread", [
    (["read"], [], True),
    ([], ["read"], True),
    (["read"], ["read"], False),
    (["read"], _MISSING, False),
    (_MISSING, [], True),
])
def test_combined_block_unread_if_any_part_unread(
    profile: Callable, first: Any, second: Any, unread: bool,
) -> None:
    profile(HUMAN)
    out = core.format_messages([msg(1, first), msg(2, second, timestamp=1060)])

    assert out.count("<msg ") == 1
    assert ('unread="true"' in out) is unread


@pytest.mark.parametrize("first, second", [
    (["read", "mentioned"], ["read"]),
    (["read"], ["read", "group_mentioned"]),
])
def test_combined_block_mentioned_if_any_part_mentioned(
    profile: Callable, first: list[str], second: list[str],
) -> None:
    profile(HUMAN)
    out = core.format_messages([msg(1, first), msg(2, second, timestamp=1060)])

    assert out.count("<msg ") == 1
    assert 'mentioned="true"' in out
    assert "unread=" not in out


def test_event_messages_carry_their_flags(profile: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    profile(HUMAN)
    first, second = msg(1), msg(2, timestamp=5000)
    events = [
        {"id": 4, "type": "message", "message": first, "flags": ["mentioned"]},
        {"id": 5, "type": "message", "message": second},
    ]
    monkeypatch.setattr(core.get_client(), "call_endpoint",
                        lambda **_: {"result": "success", "events": events}, raising=False)

    messages, _, last_event_id = core.get_events("queue", 3)

    assert [m.get("flags") for m in messages] == [["mentioned"], None]
    assert last_event_id == 5
    assert "flags" not in first
    assert 'id="1" unread="true" mentioned="true">' in core.format_messages(messages)


def test_no_config_formats_without_unread(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise FileNotFoundError("no zuliprc")

    monkeypatch.setattr(core, "get_client", missing)

    assert 'mentioned="true"' in core.format_messages([msg(1, ["mentioned"])])


def test_same_name_different_sender_is_not_combined(profile: Callable) -> None:
    profile(HUMAN)
    own, other = msg(1, [], sender_id=1), msg(2, [], timestamp=1060, sender_id=9)

    out = core.format_messages([own, other])

    assert out.count("<msg ") == 2
    assert out.count('unread="true"') == 1
