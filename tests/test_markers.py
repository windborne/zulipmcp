"""unread/mentioned markers on formatted messages."""
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pytest

from zulipmcp import core

_MISSING = object()


def msg(msg_id: int, flags: Any = _MISSING, timestamp: int = 1000) -> dict:
    m: dict = {
        "id": msg_id, "timestamp": timestamp, "type": "stream",
        "display_recipient": "general", "subject": "t", "content": "hi",
        "sender_full_name": "Alice",
    }
    if flags is not _MISSING:
        m["flags"] = flags
    return m


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], list[int]]:
    """Install a fake get_profile returning (or raising) ``result``; returns the call log."""
    monkeypatch.setattr(core, "_account_is_human", None)
    monkeypatch.setattr(core, "TIMEZONE", ZoneInfo("UTC"))

    def install(result: Any) -> list[int]:
        calls: list[int] = []

        def fake() -> dict:
            calls.append(1)
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr(core, "get_profile", fake)
        return calls

    return install


HUMAN = {"result": "success", "is_bot": False, "user_id": 1}
BOT = {"result": "success", "is_bot": True, "user_id": 2}


def tag(out: str) -> str:
    return out.splitlines()[0]


def test_human_unread(profile: Callable) -> None:
    profile(HUMAN)

    assert 'unread="true"' in core.format_messages([msg(1, [])])


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


def test_profile_fetched_once(profile: Callable) -> None:
    calls = profile(HUMAN)
    core.format_messages([msg(1, [])])
    core.format_messages([msg(2, [])])

    assert calls == [1]


@pytest.mark.parametrize("result", [
    RuntimeError("network down"),
    {"result": "error", "msg": "unauthorized"},
    {"result": "success"},
])
def test_profile_failure_means_no_unread(profile: Callable, result: Any) -> None:
    calls = profile(result)
    out = core.format_messages([msg(1, ["mentioned"]), msg(2, [], timestamp=5000)])

    assert "unread=" not in out
    assert out.count('mentioned="true"') == 1
    assert calls == [1]


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


def test_combine_leaves_inputs_untouched(profile: Callable) -> None:
    profile(HUMAN)
    first, second = msg(1, ["read"]), msg(2, [], timestamp=1060)
    core.format_messages([first, second])

    assert first["flags"] == ["read"]
    assert second["flags"] == []


def test_uncombined_messages_keep_own_markers(profile: Callable) -> None:
    profile(HUMAN)
    out = core.format_messages([msg(1, ["read"]), msg(2, [], timestamp=1060)], combine=False)
    first, second = (line for line in out.splitlines() if line.startswith("<msg "))

    assert "unread=" not in first
    assert 'unread="true"' in second
