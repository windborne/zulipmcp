import importlib
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pytest

from zulipmcp import core

# `zulipmcp.mcp` is shadowed by the package's FastMCP re-export.
mcp = importlib.import_module("zulipmcp.mcp")
get_messages = getattr(mcp.get_messages, "fn", mcp.get_messages)
search_messages = getattr(mcp.search_messages, "fn", mcp.search_messages)

BOT = "bot@example.com"
ALICE = "alice@example.com"
BOB = "bob@example.com"


class DictCache:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key: Any) -> Any:
        return self.data.get(key)

    def set(self, key: Any, value: Any, expire: Optional[int] = None) -> None:
        self.data[key] = value


class FakeClient:
    def __init__(self) -> None:
        self.email = BOT
        self.feature_level = 445
        self.messages: list[dict] = []
        self.streams: list[dict] = [
            {"name": "general", "invite_only": False},
            {"name": "secret", "invite_only": True},
            {"name": "noise", "invite_only": False},
        ]
        self.requests: list[dict] = []
        self.response: Optional[dict] = None

    def get_messages(self, request: dict) -> dict:
        if request.get("num_before") == 0 and isinstance(request.get("anchor"), int):
            hits = [m for m in self.messages if m["id"] == request["anchor"]]
            return {"result": "success", "messages": [dict(m) for m in hits]}
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return {"result": "success", "messages": [dict(m) for m in self.messages]}

    def get_streams(self, **_kwargs: Any) -> dict:
        return {"result": "success", "streams": self.streams}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(core, "get_client", lambda: fake)
    cache = DictCache()
    monkeypatch.setattr(core, "_get_cache", lambda: cache)
    monkeypatch.setattr(core, "_ignored_streams", set())
    monkeypatch.setattr(core, "TIMEZONE", ZoneInfo("UTC"))
    monkeypatch.delenv("BOT_ALLOWED_PRIVATE_STREAMS", raising=False)
    return fake


def stream_msg(msg_id: int, timestamp: int = 1000, stream: str = "general",
               topic: str = "t", content: str = "hi", sender: str = "Alice") -> dict:
    return {
        "id": msg_id, "timestamp": timestamp, "type": "stream",
        "display_recipient": stream, "subject": topic, "content": content,
        "sender_full_name": sender, "sender_email": ALICE,
    }


def dm_msg(msg_id: int, emails: list[str], timestamp: int = 1000,
           content: str = "hi") -> dict:
    return {
        "id": msg_id, "timestamp": timestamp, "type": "private",
        "display_recipient": [
            {"email": e, "full_name": e.split("@")[0], "id": i}
            for i, e in enumerate(emails)
        ],
        "subject": "", "content": content,
        "sender_full_name": "Alice", "sender_email": ALICE,
    }


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# --- narrow building ---------------------------------------------------------

@pytest.mark.parametrize("kwargs, expected", [
    ({"stream": "general", "topic": "t"},
     [{"operator": "channel", "operand": "general"}, {"operator": "topic", "operand": "t"}]),
    ({"dm_with": [ALICE, BOB]}, [{"operator": "dm", "operand": [ALICE, BOB]}]),
    ({"sender": ALICE}, [{"operator": "sender", "operand": ALICE}]),
    ({"query": "deploy plan"}, [{"operator": "search", "operand": "deploy plan"}]),
    ({"stream": "general", "sender": ALICE, "query": "x"},
     [{"operator": "channel", "operand": "general"},
      {"operator": "sender", "operand": ALICE},
      {"operator": "search", "operand": "x"}]),
    ({"dm_with": [ALICE], "query": "x"},
     [{"operator": "dm", "operand": [ALICE]}, {"operator": "search", "operand": "x"}]),
    ({"stream": "", "topic": "", "dm_with": [], "sender": "", "query": ""}, []),
])
def test_build_narrow(kwargs: dict, expected: list[dict]) -> None:
    assert core.build_narrow(**kwargs) == expected


def test_search_passes_narrow_to_server(client: FakeClient) -> None:
    search_messages(query="x", stream="general", topic="t", sender=ALICE)

    assert client.requests[0]["narrow"] == core.build_narrow(
        stream="general", topic="t", sender=ALICE, query="x",
    )


# --- anchors -----------------------------------------------------------------

def test_anchor_defaults_to_newest(client: FakeClient) -> None:
    core.fetch_narrow([], num_messages=500)

    request = client.requests[0]
    assert request["anchor"] == "newest"
    assert request["num_before"] == 100
    assert request["num_after"] == 0
    assert request["apply_markdown"] is False


def test_anchor_before_message_id_excludes_anchor(client: FakeClient) -> None:
    core.fetch_narrow([], before_message_id=123, before=utc(2025, 3, 1))

    request = client.requests[0]
    assert request["anchor"] == 123
    assert request["include_anchor"] is False
    assert "anchor_date" not in request


def test_anchor_before_date_uses_utc(client: FakeClient) -> None:
    before = datetime(2025, 3, 1, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    core.fetch_narrow([], before=before)

    request = client.requests[0]
    assert request["anchor"] == "date"
    assert request["anchor_date"] == "2025-03-01T00:00:00+00:00"
    assert request["include_anchor"] is True


def test_before_requires_feature_level(client: FakeClient) -> None:
    client.feature_level = 444

    with pytest.raises(ValueError, match=r"requires Zulip 12\.0\+"):
        core.fetch_narrow([], before=utc(2025, 3, 1))
    assert client.requests == []
    assert "requires Zulip 12.0+" in search_messages(query="x", before="2025-03-01")
    assert "requires Zulip 12.0+" in get_messages(dm_with=[ALICE], before="2025-03-01")


def test_before_filters_anchor_and_trims(client: FakeClient) -> None:
    cutoff = int(utc(2025, 3, 1).timestamp())
    client.messages = [stream_msg(i, cutoff - 10 + i) for i in range(1, 4)]
    client.messages.append(stream_msg(10, cutoff))

    msgs, _ = core.fetch_narrow([], num_messages=2, before=utc(2025, 3, 1))

    assert [m["id"] for m in msgs] == [2, 3]


def test_before_keeps_newest_anchor_when_date_is_future(client: FakeClient) -> None:
    client.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    msgs, _ = core.fetch_narrow([], num_messages=2, before=utc(2100, 1, 1))

    assert [m["id"] for m in msgs] == [1, 2]


# --- after bound -------------------------------------------------------------

def test_after_filters_and_reports_bound(client: FakeClient) -> None:
    client.messages = [stream_msg(3, 300), stream_msg(1, 100), stream_msg(2, 200)]

    msgs, reached = core.fetch_narrow([], after=datetime.fromtimestamp(200, timezone.utc))
    assert [m["id"] for m in msgs] == [2, 3]
    assert reached is True

    msgs, reached = core.fetch_narrow([], after=datetime.fromtimestamp(50, timezone.utc))
    assert [m["id"] for m in msgs] == [1, 2, 3]
    assert reached is False


# --- date parsing ------------------------------------------------------------

def test_parse_naive_and_date_only_use_configured_timezone(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokyo = ZoneInfo("Asia/Tokyo")
    monkeypatch.setattr(core, "TIMEZONE", tokyo)

    assert core.parse_datetime("2025-03-01") == datetime(2025, 3, 1, tzinfo=tokyo)
    assert core.parse_datetime("2025-03-01T14:30") == datetime(2025, 3, 1, 14, 30, tzinfo=tokyo)
    assert core.parse_datetime("2025-03-01 14:30:00") == datetime(2025, 3, 1, 14, 30, tzinfo=tokyo)


def test_parse_explicit_offsets() -> None:
    assert core.parse_datetime("2025-03-01T00:00:00Z") == utc(2025, 3, 1)
    assert core.parse_datetime("2025-03-01T09:00:00+09:00") == utc(2025, 3, 1)


@pytest.mark.parametrize("value", ["yesterday", "2025-13-01", "", "03/01/2025"])
def test_parse_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid date"):
        core.parse_datetime(value)


def test_tools_report_invalid_dates(client: FakeClient) -> None:
    assert search_messages(query="x", after="soon").startswith("Error: Invalid date")
    assert get_messages(dm_with=[ALICE], before="soon").startswith("Error: Invalid date")
    assert client.requests == []


# --- stream visibility -------------------------------------------------------

def test_private_hit_dropped_unless_allowlisted(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.messages = [stream_msg(1, stream="general"), stream_msg(2, stream="secret"),
                       dm_msg(3, [ALICE, BOT])]

    msgs, _ = core.fetch_narrow([])
    assert [m["id"] for m in msgs] == [1, 3]

    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "__ALL__")
    msgs, _ = core.fetch_narrow([])
    assert [m["id"] for m in msgs] == [1, 2, 3]


def test_ignored_streams_dropped(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_ignored_streams", {"noise"})
    client.messages = [stream_msg(1, stream="general"), stream_msg(2, stream="Noise")]

    msgs, _ = core.fetch_narrow([])

    assert [m["id"] for m in msgs] == [1]


def test_search_rejects_disallowed_private_stream(client: FakeClient) -> None:
    assert search_messages(query="x", stream="secret") == mcp.PRIVATE_STREAM_ERROR
    assert client.requests == []


def test_api_error_is_reported(client: FakeClient) -> None:
    client.response = {"result": "error", "msg": "Invalid narrow"}

    assert search_messages(sender="nobody@example.com") == "Error: Zulip API error: Invalid narrow"


# --- get_messages ------------------------------------------------------------

def test_get_messages_reads_dm(client: FakeClient) -> None:
    client.messages = [dm_msg(1, [ALICE, BOT], content="hello there")]

    out = get_messages(dm_with=[ALICE])

    assert client.requests[0]["narrow"] == [{"operator": "dm", "operand": [ALICE]}]
    assert client.requests[0]["anchor"] == "newest"
    assert out.startswith(f"Messages from DM with {ALICE}:")
    assert "hello there" in out


def test_get_messages_dm_message_id_derives_partners(client: FakeClient) -> None:
    client.messages = [dm_msg(5, [ALICE, BOB, BOT])]

    out = get_messages(message_id=5)

    request = client.requests[0]
    assert request["narrow"] == [{"operator": "dm", "operand": [ALICE, BOB]}]
    assert request["anchor"] == 6
    assert out.startswith(f"History for DM with {ALICE}, {BOB} (around message 5):")


def test_get_messages_self_dm_message_id(client: FakeClient) -> None:
    client.messages = [dm_msg(5, [BOT])]

    get_messages(message_id=5)

    assert client.requests[0]["narrow"] == [{"operator": "dm", "operand": [BOT]}]


def test_get_messages_unknown_message_id(client: FakeClient) -> None:
    assert get_messages(message_id=99) == "Message 99 not found or not accessible."


def test_get_messages_stream_without_dates_uses_topic_fetch(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        core, "get_topic_messages",
        lambda stream, topic, num_messages=20, before_message_id=None:
            calls.append((stream, topic, num_messages, before_message_id)) or [stream_msg(1)],
    )

    out = get_messages(stream="general", topic="t", before_message_id=50)

    assert calls == [("general", "t", 20, 50)]
    assert out.startswith("Messages from [public] #general > t:")


def test_get_messages_stream_with_after(client: FakeClient) -> None:
    client.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    out = get_messages(stream="general", topic="t", after="1970-01-01T00:02:00")

    assert client.requests[0]["narrow"] == core.build_narrow(stream="general", topic="t")
    assert 'id="2"' in out and 'id="1"' not in out
    assert "Reached after=1970-01-01T00:02:00" in out


@pytest.mark.parametrize("kwargs, error", [
    ({}, "Error: Provide stream+topic, dm_with, or message_id."),
    ({"stream": "general"}, "Error: Provide stream+topic, dm_with, or message_id."),
    ({"dm_with": [ALICE], "stream": "general", "topic": "t"},
     "Error: Provide stream+topic or dm_with, not both."),
])
def test_get_messages_target_validation(client: FakeClient, kwargs: dict, error: str) -> None:
    assert get_messages(**kwargs) == error


# --- search_messages ---------------------------------------------------------

@pytest.mark.parametrize("kwargs, error", [
    ({}, "Error: Provide at least one of query, stream, topic, sender, or dm_with."),
    ({"after": "2025-03-01"}, "Error: Provide at least one of query, stream, topic, sender, or dm_with."),
    ({"topic": "t"}, "Error: topic requires stream."),
    ({"dm_with": [ALICE], "stream": "general"}, "Error: dm_with cannot be combined with stream/topic."),
])
def test_search_validation(client: FakeClient, kwargs: dict, error: str) -> None:
    assert search_messages(**kwargs) == error
    assert client.requests == []


def test_search_header_and_output(client: FakeClient) -> None:
    client.messages = [
        stream_msg(7, 100, content="deploy now"),
        dm_msg(9, [ALICE, BOT], timestamp=110, content="deploy later"),
        stream_msg(8, 105, content="/nb deploy secret"),
    ]

    out = search_messages(query="deploy")

    assert out.startswith(f"Search as {BOT} (2 results; older: before_message_id=7)\n\n")
    assert 'stream="general"' in out and 'topic="t"' in out
    assert f'dm="{ALICE}, {BOT}"' in out
    assert "deploy secret" not in out


def test_search_header_reports_after_bound(client: FakeClient) -> None:
    client.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    out = search_messages(query="x", after="1970-01-01T00:02:00")

    assert out.startswith(f"Search as {BOT} (1 result; reached after=1970-01-01T00:02:00)")


def test_search_header_without_results(client: FakeClient) -> None:
    assert search_messages(query="x") == f"Search as {BOT} (0 results)"


def test_format_messages_dm_attribute() -> None:
    out = core.format_messages([dm_msg(1, [ALICE, BOT])], include_topic=True)

    assert f'dm="{ALICE}, {BOT}"' in out
    assert "stream=" not in out
    assert "dm=" not in core.format_messages([dm_msg(1, [ALICE, BOT])])
