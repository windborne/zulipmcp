import asyncio
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
ME = 100
PUBLIC = [{"operator": "channels", "operand": "public"}]


class DictCache:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key: Any) -> Any:
        return self.data.get(key)

    def set(self, key: Any, value: Any, expire: Optional[int] = None) -> None:
        self.data[key] = value


def _dm_operand_error(narrow: list[dict]) -> Optional[str]:
    """Mirror the server's check_string_or_int_list validation of ``dm``."""
    for term in narrow:
        if term["operator"] != "dm":
            continue
        operand = term["operand"]
        if isinstance(operand, str):
            continue
        for i, item in enumerate(operand):
            if not isinstance(item, int):
                return f"operand[{i}] is not an integer"
    return None


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
        self.single_reads: list[tuple[str, dict]] = []
        self.response: Optional[dict] = None

    def call_endpoint(self, url: str, method: str, request: dict) -> dict:
        assert method == "GET"
        self.single_reads.append((url, request))
        hits = [m for m in self.messages if url == f"/messages/{m['id']}"]
        if not hits:
            return {"result": "error", "msg": "Invalid message(s)"}
        return {"result": "success", "message": dict(hits[0])}

    def get_profile(self) -> dict:
        return {"result": "success", "is_bot": False, "user_id": ME}

    def get_messages(self, request: dict) -> dict:
        self.requests.append(request)
        error = _dm_operand_error(request.get("narrow", []))
        if error:
            return {"result": "error", "msg": error}
        if self.response is not None:
            return self.response
        narrow = request.get("narrow", [])
        known = {s["name"].lower() for s in self.streams}
        for term in narrow:
            if term["operator"] in ("channel", "stream") and term["operand"].lower() not in known:
                return {"result": "error", "msg": f"Invalid narrow operator: unknown channel {term['operand']}"}
        msgs = sorted((m for m in self.messages if self._in_scope(m, narrow)),
                      key=lambda m: m["id"])
        anchor = request.get("anchor")
        if anchor == "date":
            page, found_oldest = msgs, True
        else:
            if isinstance(anchor, int):
                inclusive = request.get("include_anchor", True)
                msgs = [m for m in msgs if m["id"] < anchor or (inclusive and m["id"] == anchor)]
            limit = request["num_before"]
            page = msgs[-limit:] if limit else []
            found_oldest = len(page) == len(msgs)
        return {"result": "success", "messages": [dict(m) for m in page],
                "found_oldest": found_oldest}

    def _in_scope(self, msg: dict, narrow: list[dict]) -> bool:
        """Apply the narrow's channel terms, as the server would."""
        private = {s["name"].lower() for s in self.streams if s["invite_only"]}
        stream = msg["display_recipient"].lower() if msg["type"] == "stream" else None
        for term in narrow:
            operator, operand = term["operator"], term["operand"]
            if operator in ("channel", "stream") and stream != operand.lower():
                return False
            if operator in ("channels", "streams") and (stream is None or stream in private):
                return False
        return True

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


@pytest.fixture
def dm_reads(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "__ALL__")
    return client


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
    ({"dm_with": [ALICE, BOB]}, [{"operator": "dm", "operand": f"{ALICE},{BOB}"}]),
    ({"sender": ALICE}, [{"operator": "sender", "operand": ALICE}]),
    ({"query": "deploy plan"}, [{"operator": "search", "operand": "deploy plan"}]),
    ({"stream": "general", "sender": ALICE, "query": "x"},
     [{"operator": "channel", "operand": "general"},
      {"operator": "sender", "operand": ALICE},
      {"operator": "search", "operand": "x"}]),
    ({"dm_with": [ALICE], "query": "x"},
     [{"operator": "dm", "operand": ALICE}, {"operator": "search", "operand": "x"}]),
    ({"stream": "", "topic": "", "dm_with": [], "sender": "", "query": "", "only": [],
      "dm_including": []}, []),
    ({"only": ["unread", "mentioned"]},
     [{"operator": "is", "operand": "unread"}, {"operator": "is", "operand": "mentioned"}]),
    ({"dm_including": [ALICE, BOB]},
     [{"operator": "dm-including", "operand": ALICE},
      {"operator": "dm-including", "operand": BOB}]),
    ({"stream": "general", "topic": "t", "only": ["starred"]},
     [{"operator": "channel", "operand": "general"}, {"operator": "topic", "operand": "t"},
      {"operator": "is", "operand": "starred"}]),
    ({"dm_including": [ALICE], "query": "x", "only": ["unread"]},
     [{"operator": "dm-including", "operand": ALICE}, {"operator": "search", "operand": "x"},
      {"operator": "is", "operand": "unread"}]),
])
def test_build_narrow(kwargs: dict, expected: list[dict]) -> None:
    assert core.build_narrow(**kwargs) == expected


def test_fake_server_rejects_email_list_dm_operand(dm_reads: FakeClient) -> None:
    with pytest.raises(ValueError, match=r"operand\[0\] is not an integer"):
        core.fetch_narrow([{"operator": "dm", "operand": [ALICE]}])
    core.fetch_narrow([{"operator": "dm", "operand": [7, 8]}])


def test_dm_narrows_pass_server_validation(dm_reads: FakeClient) -> None:
    dm_reads.messages = [dm_msg(5, [ALICE, BOB, BOT], content="group hello")]

    assert "group hello" in get_messages(dm_with=[ALICE, BOB])
    assert "group hello" in get_messages(message_id=5)
    assert "group hello" in search_messages(query="hello", dm_with=[ALICE, BOB])


def test_search_passes_narrow_to_server(client: FakeClient) -> None:
    search_messages(query="x", stream="general", topic="t", sender=ALICE)

    assert client.requests[0]["narrow"] == core.build_narrow(
        stream="general", topic="t", sender=ALICE, query="x",
    )


# --- anchors -----------------------------------------------------------------

def test_anchor_defaults_to_newest(dm_reads: FakeClient) -> None:
    core.fetch_narrow([], num_messages=500)

    request = dm_reads.requests[0]
    assert request["anchor"] == "newest"
    assert request["num_before"] == 100
    assert request["num_after"] == 0
    assert request["apply_markdown"] is False


@pytest.mark.parametrize("num_messages, expected", [(0, 1), (-5, 1), (500, 100)])
def test_num_messages_is_clamped(dm_reads: FakeClient, num_messages: int, expected: int) -> None:
    dm_reads.messages = [stream_msg(1), stream_msg(2)]

    page = core.fetch_narrow([], num_messages=num_messages)

    assert dm_reads.requests[0]["num_before"] == expected
    assert [m["id"] for m in page.messages] == [1, 2][-expected:]


def test_anchor_before_message_id_excludes_anchor(dm_reads: FakeClient) -> None:
    core.fetch_narrow([], before_message_id=123, before=utc(2025, 3, 1))

    request = dm_reads.requests[0]
    assert request["anchor"] == 123
    assert request["include_anchor"] is False
    assert "anchor_date" not in request


def test_anchor_before_date_uses_utc(dm_reads: FakeClient) -> None:
    before = datetime(2025, 3, 1, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    core.fetch_narrow([], before=before)

    request = dm_reads.requests[0]
    assert request["anchor"] == "date"
    assert request["anchor_date"] == "2025-03-01T00:00:00+00:00"
    assert request["include_anchor"] is True


def test_before_requires_feature_level(dm_reads: FakeClient) -> None:
    dm_reads.feature_level = 444

    with pytest.raises(ValueError, match=r"requires Zulip 12\.0\+"):
        core.fetch_narrow([], before=utc(2025, 3, 1))
    assert dm_reads.requests == []
    assert "requires Zulip 12.0+" in search_messages(query="x", before="2025-03-01")
    assert "requires Zulip 12.0+" in get_messages(stream="general", topic="t", before="2025-03-01")


def test_before_filters_anchor_and_trims(dm_reads: FakeClient) -> None:
    cutoff = int(utc(2025, 3, 1).timestamp())
    dm_reads.messages = [stream_msg(i, cutoff - 10 + i) for i in range(1, 4)]
    dm_reads.messages.append(stream_msg(10, cutoff))

    page = core.fetch_narrow([], num_messages=2, before=utc(2025, 3, 1))

    assert [m["id"] for m in page.messages] == [2, 3]
    assert page.older_than == 2


def test_before_keeps_newest_anchor_when_date_is_future(dm_reads: FakeClient) -> None:
    dm_reads.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    page = core.fetch_narrow([], num_messages=2, before=utc(2100, 1, 1))

    assert [m["id"] for m in page.messages] == [1, 2]
    assert page.older_than is None


# --- after bound -------------------------------------------------------------

def test_after_filters_and_stops_paging(dm_reads: FakeClient) -> None:
    dm_reads.messages = [stream_msg(3, 300), stream_msg(1, 100), stream_msg(2, 200)]

    page = core.fetch_narrow([], num_messages=3, after=datetime.fromtimestamp(200, timezone.utc))
    assert [m["id"] for m in page.messages] == [2, 3]
    assert page.older_than is None

    page = core.fetch_narrow([], num_messages=2, after=datetime.fromtimestamp(50, timezone.utc))
    assert [m["id"] for m in page.messages] == [2, 3]
    assert page.older_than == 2


def test_naive_bounds_use_configured_timezone(
    dm_reads: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "TIMEZONE", ZoneInfo("Asia/Tokyo"))
    cutoff = int(utc(2025, 3, 1).timestamp())
    dm_reads.messages = [stream_msg(1, cutoff - 3600), stream_msg(2, cutoff - 1),
                         stream_msg(3, cutoff)]

    page = core.fetch_narrow([], after=datetime(2025, 3, 1, 8, 0),
                             before=datetime(2025, 3, 1, 9, 0))

    assert dm_reads.requests[0]["anchor_date"] == "2025-03-01T00:00:00+00:00"
    assert [m["id"] for m in page.messages] == [1, 2]


def test_naive_and_aware_bounds_mix(dm_reads: FakeClient) -> None:
    core.fetch_narrow([], after=datetime(2025, 3, 1), before=utc(2025, 3, 2))

    with pytest.raises(ValueError, match="after must be earlier than before"):
        core.fetch_narrow([], after=datetime(2025, 3, 2), before=utc(2025, 3, 1))


def test_bounds_compare_whole_seconds(dm_reads: FakeClient) -> None:
    # Zulip timestamps are whole seconds, so a message stamped 200 may have
    # been sent after 200.5.
    dm_reads.messages = [stream_msg(1, 199), stream_msg(2, 200), stream_msg(3, 201),
                         stream_msg(4, 202)]

    page = core.fetch_narrow([], after=datetime.fromtimestamp(200.5, timezone.utc),
                             before=datetime.fromtimestamp(201.5, timezone.utc))

    assert [m["id"] for m in page.messages] == [2, 3]


def test_after_must_precede_before(dm_reads: FakeClient) -> None:
    with pytest.raises(ValueError, match="after must be earlier than before"):
        core.fetch_narrow([], after=utc(2025, 3, 1), before=utc(2025, 3, 1))
    error = "Error: after must be earlier than before"
    assert search_messages(query="x", after="2025-03-02", before="2025-03-01") == error
    assert get_messages(stream="general", topic="t", after="2025-03-02", before="2025-03-01") == error
    assert dm_reads.requests == []


# --- paging ------------------------------------------------------------------

def test_older_than_pages_back_until_oldest(dm_reads: FakeClient) -> None:
    dm_reads.messages = [stream_msg(i) for i in range(1, 6)]

    pages = []
    cursor = None
    while True:
        page = core.fetch_narrow([], num_messages=2, before_message_id=cursor)
        pages.append([m["id"] for m in page.messages])
        if page.older_than is None:
            break
        cursor = page.older_than

    assert pages == [[4, 5], [2, 3], [1]]
    assert dm_reads.requests[1]["include_anchor"] is False


def test_older_than_counts_messages_dropped_client_side(client: FakeClient) -> None:
    # A server page that still holds a hidden message: it is dropped, but
    # the cursor is computed from the server's page.
    client.response = {"result": "success", "found_oldest": False, "messages": [
        stream_msg(1), stream_msg(2, stream="secret"), stream_msg(3)]}

    page = core.fetch_narrow(PUBLIC, num_messages=2)

    assert [m["id"] for m in page.messages] == [3]
    assert page.older_than == 2


def test_older_than_is_none_for_empty_page(dm_reads: FakeClient) -> None:
    assert core.fetch_narrow([]) == core.MessagePage([], None)


# --- date parsing ------------------------------------------------------------

def test_parse_naive_and_date_only_use_configured_timezone(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokyo = ZoneInfo("Asia/Tokyo")
    monkeypatch.setattr(core, "TIMEZONE", tokyo)

    assert mcp._parse_when("2025-03-01") == datetime(2025, 3, 1, tzinfo=tokyo)
    assert mcp._parse_when("2025-03-01T14:30") == datetime(2025, 3, 1, 14, 30, tzinfo=tokyo)
    assert mcp._parse_when("2025-03-01 14:30:00") == datetime(2025, 3, 1, 14, 30, tzinfo=tokyo)


def test_parse_explicit_offsets() -> None:
    assert mcp._parse_when("2025-03-01T00:00:00Z") == utc(2025, 3, 1)
    assert mcp._parse_when("2025-03-01T09:00:00+09:00") == utc(2025, 3, 1)


def test_parse_empty_is_unbounded() -> None:
    assert mcp._parse_when("") is None


@pytest.mark.parametrize("value", ["yesterday", "2025-13-01", " ", "03/01/2025"])
def test_parse_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid date"):
        mcp._parse_when(value)


def test_tools_report_invalid_dates(client: FakeClient) -> None:
    assert search_messages(query="x", after="soon").startswith("Error: Invalid date")
    assert get_messages(stream="general", topic="t", before="soon").startswith("Error: Invalid date")
    assert client.requests == []


# --- read gate ---------------------------------------------------------------

UNSCOPED = "Reading without naming a channel requires BOT_ALLOWED_PRIVATE_STREAMS=__ALL__"


DM_READS_ERROR = f"Error: {core.DM_READS_DENIED}"


def test_error_strings() -> None:
    assert mcp.PRIVATE_STREAM_ERROR == (
        "Error: Private stream access denied: not in BOT_ALLOWED_PRIVATE_STREAMS."
    )
    assert DM_READS_ERROR == "Error: Reading DMs requires BOT_ALLOWED_PRIVATE_STREAMS=__ALL__."


@pytest.mark.parametrize("narrow", [
    [],
    [{"operator": "search", "operand": "x"}],
    [{"operator": "channels", "operand": "archived"}],
    [{"operator": "channels", "operand": "public", "negated": True}],
    [{"operator": "channel", "operand": "general", "negated": True}],
])
def test_gate_rejects_unscoped_narrow(client: FakeClient, narrow: list[dict]) -> None:
    with pytest.raises(core.ReadScopeError, match=UNSCOPED) as exc:
        core.fetch_narrow(narrow)
    assert "Allowlisted" not in str(exc.value)
    assert client.requests == []


def test_gate_unscoped_error_lists_allowlist(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "Secret,ops")

    with pytest.raises(ValueError, match="Allowlisted private channels: ops, secret."):
        core.fetch_narrow([])
    assert client.requests == []


@pytest.mark.parametrize("term", [
    {"operator": "dm", "operand": ALICE},
    {"operator": "dm-including", "operand": ALICE},
    {"operator": "pm-with", "operand": ALICE},
    {"operator": "pm_with", "operand": ALICE},
    {"operator": "group-pm-with", "operand": ALICE},
    {"operator": "group_pm_with", "operand": ALICE},
    {"operator": "is", "operand": "dm"},
    {"operator": "is", "operand": "private"},
    {"operator": "is", "operand": "dm", "negated": "false"},
    {"operator": "dm", "operand": ALICE, "negated": 0},
])
def test_gate_rejects_dm_narrow(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, term: dict,
) -> None:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "secret")

    with pytest.raises(core.ReadScopeError) as exc:
        core.fetch_narrow([*PUBLIC, term])
    assert f"Error: {exc.value}" == DM_READS_ERROR
    assert client.requests == []


@pytest.mark.parametrize("operator", ["channel", "stream"])
def test_gate_rejects_disallowed_private_channel(client: FakeClient, operator: str) -> None:
    with pytest.raises(core.ReadScopeError) as exc:
        core.fetch_narrow([{"operator": operator, "operand": "secret"}])
    assert f"Error: {exc.value}" == mcp.PRIVATE_STREAM_ERROR
    assert client.requests == []


@pytest.mark.parametrize("narrow", [
    [["channel", "general"]],
    [*PUBLIC, ["dm", ALICE]],
    [*PUBLIC, "is:dm"],
    [*PUBLIC, None],
])
def test_gate_rejects_malformed_terms(client: FakeClient, narrow: list) -> None:
    with pytest.raises(core.ReadScopeError, match="must be dicts"):
        core.fetch_narrow(narrow)
    assert client.requests == []


@pytest.mark.parametrize("narrow", [
    [{"operator": "channel", "operand": "general"}, {"operator": "with", "operand": 9001}],
    [*PUBLIC, {"operator": "with", "operand": "9001"}],
])
def test_gate_rejects_with_operator(client: FakeClient, narrow: list[dict]) -> None:
    with pytest.raises(core.ReadScopeError, match="with operator requires"):
        core.fetch_narrow(narrow)
    assert client.requests == []


@pytest.mark.parametrize("operator", ["channel", "stream"])
@pytest.mark.parametrize("operand", [7, 7.0, ["general"], None])
def test_gate_rejects_non_name_channel_operand(
    client: FakeClient, operator: str, operand: Any,
) -> None:
    with pytest.raises(core.ReadScopeError, match="Channel operands must be names"):
        core.fetch_narrow([*PUBLIC, {"operator": operator, "operand": operand}])
    assert client.requests == []


def test_gate_ignores_negated_with_and_dm_terms(client: FakeClient) -> None:
    narrow = [*PUBLIC, {"operator": "with", "operand": 1, "negated": True},
              {"operator": "dm", "operand": ALICE, "negated": True}]

    core.fetch_narrow(narrow)

    assert client.requests[0]["narrow"] == narrow


def test_read_scope_error_is_value_error() -> None:
    assert issubclass(core.ReadScopeError, ValueError)


def test_gate_fails_closed_when_stream_list_errors(client: FakeClient) -> None:
    client.get_streams = lambda **_kwargs: {"result": "error", "msg": "boom"}

    with pytest.raises(ValueError, match="Private stream access denied"):
        core.fetch_narrow(core.build_narrow(stream="general"))
    assert client.requests == []


@pytest.mark.parametrize("allowlist, narrow", [
    (None, core.build_narrow(stream="general", query="x")),
    (None, [{"operator": "stream", "operand": "general"}]),
    ("secret", core.build_narrow(stream="secret", topic="t")),
    (None, PUBLIC + [{"operator": "is", "operand": "dm", "negated": True}]),
    (None, [{"operator": "streams", "operand": "public"}]),
    (None, [{"operator": "channels", "operand": "web-public"}]),
    ("__ALL__", []),
    ("__ALL__", core.build_narrow(dm_with=[ALICE], query="x")),
    ("__ALL__", core.build_narrow(stream="secret")),
    ("__ALL__", [{"operator": "channels", "operand": "archived"}]),
])
def test_gate_sends_allowed_narrow_unmodified(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
    allowlist: Optional[str], narrow: list[dict],
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)

    core.fetch_narrow(narrow)

    assert client.requests[0]["narrow"] == narrow


def test_private_and_dm_hits_follow_allowlist(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.messages = [stream_msg(1, stream="general"), stream_msg(2, stream="secret"),
                       dm_msg(3, [ALICE, BOT])]

    assert [m["id"] for m in core.fetch_narrow(PUBLIC).messages] == [1]

    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "secret")
    assert [m["id"] for m in core.fetch_narrow(PUBLIC).messages] == [1]
    assert [m["id"] for m in core.fetch_narrow(core.build_narrow(stream="secret")).messages] == [2]

    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "__ALL__")
    assert [m["id"] for m in core.fetch_narrow([]).messages] == [1, 2, 3]


def test_ignored_streams_dropped(dm_reads: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_ignored_streams", {"noise"})
    dm_reads.messages = [stream_msg(1, stream="general"),
                         stream_msg(2, stream="Noise", content="drowned out"),
                         dm_msg(3, [ALICE, BOT])]

    assert [m["id"] for m in core.fetch_narrow([]).messages] == [1, 3]
    assert [m["id"] for m in core.fetch_narrow(PUBLIC).messages] == [1]
    assert "drowned out" not in search_messages(query="x")
    assert all(r["narrow"] in ([], PUBLIC, [{"operator": "search", "operand": "x"}])
               for r in dm_reads.requests)


def test_ignored_stream_reads_when_named(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_ignored_streams", {"noise"})
    client.messages = [stream_msg(2, stream="noise", content="drowned out")]

    assert "drowned out" in search_messages(query="x", stream="noise")
    assert "drowned out" in get_messages(stream="noise", topic="t")
    assert [m["id"] for m in core.get_topic_messages("noise", "t")] == [2]


@pytest.mark.parametrize("num_messages", [0, 1, 20])
def test_hidden_message_ids_never_surface(client: FakeClient, num_messages: int) -> None:
    client.messages = [stream_msg(9001, stream="secret", content="hidden"),
                       dm_msg(9002, [ALICE, BOT], content="hidden")]

    page = core.fetch_narrow(PUBLIC, num_messages=num_messages)
    out = search_messages(query="x", num_messages=num_messages)

    assert page == core.MessagePage([], None)
    assert "900" not in out and "hidden" not in out


def test_read_tools_reject_disallowed_private_stream(client: FakeClient) -> None:
    assert search_messages(query="x", stream="secret") == mcp.PRIVATE_STREAM_ERROR
    assert get_messages(stream="secret", topic="t") == mcp.PRIVATE_STREAM_ERROR
    assert client.requests == []


def test_api_error_is_reported(client: FakeClient) -> None:
    client.response = {"result": "error", "msg": "Invalid narrow"}

    assert search_messages(sender="nobody@example.com") == "Error: Zulip API error: Invalid narrow"


# --- get_messages ------------------------------------------------------------

def test_get_messages_reads_dm(dm_reads: FakeClient) -> None:
    dm_reads.messages = [dm_msg(1, [ALICE, BOT], content="hello there")]

    out = get_messages(dm_with=[ALICE])

    assert dm_reads.requests[0]["narrow"] == [{"operator": "dm", "operand": ALICE}]
    assert dm_reads.requests[0]["anchor"] == "newest"
    assert out.startswith(f"Messages from DM with {ALICE}:")
    assert "hello there" in out


def test_get_messages_dm_message_id_derives_partners(dm_reads: FakeClient) -> None:
    dm_reads.messages = [dm_msg(5, [ALICE, BOB, BOT])]

    out = get_messages(message_id=5)

    request = dm_reads.requests[0]
    assert request["narrow"] == [{"operator": "dm", "operand": f"{ALICE},{BOB},{BOT}"}]
    assert (request["anchor"], request["include_anchor"]) == (6, False)
    assert out.startswith(f"History for DM with {ALICE}, {BOB} (up to message 5):")


def test_get_messages_self_dm_message_id(dm_reads: FakeClient) -> None:
    dm_reads.messages = [dm_msg(5, [BOT])]

    out = get_messages(message_id=5)

    assert dm_reads.requests[0]["narrow"] == [{"operator": "dm", "operand": BOT}]
    assert out.startswith(f"History for DM with {BOT} (up to message 5):")


def test_get_messages_unknown_message_id(client: FakeClient) -> None:
    assert get_messages(message_id=99) == "Message 99 not found or not accessible."


def test_get_messages_before_message_id_is_exclusive(client: FakeClient) -> None:
    client.messages = [stream_msg(i) for i in range(1, 6)]

    out = get_messages(stream="general", topic="t", before_message_id=4, num_messages=2)

    request = client.requests[0]
    assert request["narrow"] == core.build_narrow(stream="general", topic="t")
    assert (request["anchor"], request["include_anchor"]) == (4, False)
    assert out.startswith("Messages from [public] #general > t:\n(Older: before_message_id=2)")
    assert 'id="3"' in out and 'id="4"' not in out


def test_get_messages_stream_message_id_includes_it(client: FakeClient) -> None:
    client.messages = [stream_msg(i) for i in range(1, 6)]

    out = get_messages(message_id=3)

    assert (client.requests[0]["anchor"], client.requests[0]["include_anchor"]) == (4, False)
    assert out.startswith("History for [public] #general > t (up to message 3):\n(No older messages.)")
    assert 'id="3"' in out and 'id="4"' not in out


def test_get_messages_message_id_with_before_message_id(client: FakeClient) -> None:
    client.messages = [stream_msg(i) for i in range(1, 6)]

    out = get_messages(message_id=5, before_message_id=3)

    assert (client.requests[0]["anchor"], client.requests[0]["include_anchor"]) == (3, False)
    assert out.startswith("Messages from [public] #general > t:")


def test_get_messages_message_id_in_disallowed_private_stream(client: FakeClient) -> None:
    client.messages = [stream_msg(5, stream="secret")]

    assert get_messages(message_id=5) == "Message 5 not found or not accessible."
    assert client.requests == []


def test_get_message_by_id_returns_historical_message(client: FakeClient) -> None:
    client.messages = [{**stream_msg(5, content="from before"), "flags": ["read", "historical"]}]

    msg = core.get_message_by_id(5)

    assert (msg["content"], msg["flags"]) == ("from before", ["read", "historical"])
    assert client.single_reads == [("/messages/5", {"apply_markdown": False})]
    assert core.get_message_by_id(6) is None


def test_discover_message_context_hides_unreadable_stream(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.messages = [stream_msg(5, stream="secret", topic="plans")]

    assert core.discover_message_context(5) is None

    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "secret")
    assert core.discover_message_context(5) == ("secret", "plans")


def test_get_topic_messages_is_inclusive(client: FakeClient) -> None:
    client.messages = [stream_msg(i) for i in range(1, 6)]

    msgs = core.get_topic_messages("general", "t", num_messages=2, before_message_id=4)

    assert [m["id"] for m in msgs] == [3, 4]
    assert client.requests[0]["narrow"] == core.build_narrow(stream="general", topic="t")
    assert (client.requests[0]["anchor"], client.requests[0]["include_anchor"]) == (5, False)
    assert [m["id"] for m in core.get_topic_messages("general", "t")] == [1, 2, 3, 4, 5]


def test_get_topic_messages_empty_on_error_or_private(client: FakeClient) -> None:
    client.messages = [stream_msg(1, stream="secret")]
    assert core.get_topic_messages("secret", "t") == []
    assert client.requests == []

    client.response = {"result": "error", "msg": "boom"}
    assert core.get_topic_messages("general", "t") == []


def test_get_messages_stream_with_after(client: FakeClient) -> None:
    client.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    out = get_messages(stream="general", topic="t", after="1970-01-01T00:02:00")

    assert client.requests[0]["narrow"] == core.build_narrow(stream="general", topic="t")
    assert 'id="2"' in out and 'id="1"' not in out
    assert "(No older messages at or after 1970-01-01T00:02:00.)" in out


@pytest.mark.parametrize("kwargs, error", [
    ({}, "Error: Provide stream+topic, dm_with, or message_id."),
    ({"stream": "general"}, "Error: Provide stream+topic, dm_with, or message_id."),
    ({"dm_with": [ALICE], "stream": "general", "topic": "t"},
     "Error: Provide stream+topic or dm_with, not both."),
])
def test_get_messages_target_validation(client: FakeClient, kwargs: dict, error: str) -> None:
    assert get_messages(**kwargs) == error


# --- DM reads ----------------------------------------------------------------

@pytest.mark.parametrize("allowlist", [None, "secret", '["general", "secret"]'])
def test_dm_reads_denied_without_allow_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, allowlist: Optional[str],
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)
    client.messages = [stream_msg(4, content="public hello"),
                       dm_msg(5, [ALICE, BOT], content="private hello")]

    assert not core.dm_reads_allowed()
    assert get_messages(dm_with=[ALICE]) == DM_READS_ERROR
    assert get_messages(message_id=5) == "Message 5 not found or not accessible."
    assert search_messages(query="hello", dm_with=[ALICE]) == DM_READS_ERROR
    assert client.requests == []

    out = search_messages(query="hello")
    assert "public hello" in out and "private hello" not in out
    assert "__ALL__" in DM_READS_ERROR


@pytest.mark.parametrize("allowlist", ["__ALL__", '["__ALL__"]'])
def test_dm_reads_allowed_with_allow_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, allowlist: str,
) -> None:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)
    client.messages = [dm_msg(5, [ALICE, BOT], content="private hello")]

    assert core.dm_reads_allowed()
    assert "private hello" in get_messages(dm_with=[ALICE])
    assert "private hello" in get_messages(message_id=5)
    assert "private hello" in search_messages(query="hello", dm_with=[ALICE])
    assert "private hello" in search_messages(query="hello")


def _tool(name: str):
    fn = getattr(mcp, name)
    return getattr(fn, "fn", fn)


def test_single_message_tools_deny_dms_without_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.messages = [dm_msg(5, [ALICE, BOT], content="private hello")]

    assert core.get_message_by_id(5) is None
    assert "private hello" not in _tool("get_message_by_id")(5)
    assert "private hello" not in _tool("verify_message")(5)
    assert _tool("get_message_link")(5) == "Message 5 not found or not accessible."


def test_single_message_tools_allow_dms_with_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", "__ALL__")
    client.messages = [dm_msg(5, [ALICE, BOT], content="private hello")]

    assert "private hello" in _tool("get_message_by_id")(5)
    assert "private hello" in _tool("verify_message")(5)
    monkeypatch.setattr(client, "base_url", "https://zulip.example.com/api/", raising=False)
    assert _tool("get_message_link")(5) == "[DM](https://zulip.example.com/#narrow/id/5)"


# --- search_messages ---------------------------------------------------------

NO_SEARCH_TERMS = ("Error: Provide at least one of query, stream, sender, dm_with, "
                   "dm_including, or only.")
DM_WITH_STREAM = "Error: dm_with, dm_including and only=['dm'] cannot be combined with stream/topic."
DM_WITH_PUBLIC = "Error: scope='public' cannot be combined with dm_with, dm_including or only=['dm']."

@pytest.mark.parametrize("kwargs, error", [
    ({}, NO_SEARCH_TERMS),
    ({"after": "2025-03-01"}, NO_SEARCH_TERMS),
    ({"only": []}, NO_SEARCH_TERMS),
    ({"topic": "t"}, "Error: topic requires stream."),
    ({"dm_with": [ALICE], "stream": "general"}, DM_WITH_STREAM),
    ({"dm_including": [ALICE], "stream": "general", "topic": "t"}, DM_WITH_STREAM),
    ({"only": ["dm"], "stream": "general"}, DM_WITH_STREAM),
    ({"only": ["unread", "dm"], "scope": "public"}, DM_WITH_PUBLIC),
    ({"dm_including": [ALICE], "scope": "public"}, DM_WITH_PUBLIC),
])
def test_search_validation(client: FakeClient, kwargs: dict, error: str) -> None:
    assert search_messages(**kwargs) == error
    assert client.requests == []


def test_search_scope_received_adds_no_channel_operator(dm_reads: FakeClient) -> None:
    search_messages(query="x")
    search_messages(query="x", scope="received")

    assert [r["narrow"] for r in dm_reads.requests] == [[{"operator": "search", "operand": "x"}]] * 2


@pytest.mark.parametrize("allowlist", [None, "secret"])
def test_search_defaults_to_public_without_allow_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, allowlist: Optional[str],
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)

    search_messages(query="x")
    search_messages(query="x", stream="general")

    assert [r["narrow"] for r in client.requests] == [
        PUBLIC + [{"operator": "search", "operand": "x"}],
        core.build_narrow(stream="general", query="x"),
    ]


def test_search_explicit_received_needs_allow_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", '["secret", "ops"]')

    out = search_messages(query="x", scope="received")

    assert out.startswith(f"Error: {UNSCOPED}")
    assert out.endswith("Allowlisted private channels: ops, secret.")
    assert client.requests == []


@pytest.mark.parametrize("allowlist, kwargs, hint", [
    ("secret,ops", {}, True),
    ("secret,ops", {"scope": "public"}, True),
    ("secret,ops", {"stream": "general"}, False),
    ("secret,ops", {"stream": "general", "scope": "public"}, False),
    (None, {}, False),
    ("__ALL__", {}, False),
    ("__ALL__", {"scope": "public"}, False),
])
def test_search_header_hints_allowlisted_streams(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
    allowlist: Optional[str], kwargs: dict, hint: bool,
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)

    header = search_messages(query="x", **kwargs).split("\n")[0]

    assert header.endswith("; also searchable: stream=ops, stream=secret)") == hint
    assert ("also searchable" in header) == hint


def test_search_scope_public_searches_all_public_channels(client: FakeClient) -> None:
    search_messages(query="x", sender=ALICE, scope="public")

    assert client.requests[0]["narrow"] == [
        {"operator": "channels", "operand": "public"},
        {"operator": "sender", "operand": ALICE},
        {"operator": "search", "operand": "x"},
    ]


def test_search_scope_public_conflicts_with_dm_with(dm_reads: FakeClient) -> None:
    assert search_messages(query="x", scope="public", dm_with=[ALICE]) == DM_WITH_PUBLIC
    assert dm_reads.requests == []


def test_search_stream_overrides_scope_public(dm_reads: FakeClient) -> None:
    search_messages(query="x", stream="general", scope="public")

    assert dm_reads.requests[0]["narrow"] == core.build_narrow(stream="general", query="x")


def test_search_header_and_output(dm_reads: FakeClient) -> None:
    dm_reads.messages = [
        stream_msg(7, 100, content="deploy now"),
        dm_msg(9, [ALICE, BOT], timestamp=110, content="deploy later"),
        stream_msg(8, 105, content="/nb deploy secret"),
    ]

    out = search_messages(query="deploy")

    assert out.startswith(f"Search as {BOT} (2 results; no older matches)\n\n")
    assert 'stream="general"' in out and 'topic="t"' in out
    assert f'dm="{ALICE}, {BOT}"' in out
    assert "deploy secret" not in out


def test_search_tags_unread_and_mentioned(dm_reads: FakeClient) -> None:
    dm_reads.messages = [
        {**stream_msg(1, 100), "flags": ["mentioned"]},
        {**stream_msg(2, 105), "flags": ["read"]},
        {**dm_msg(3, [ALICE, BOT], timestamp=110), "flags": [], "sender_id": ME},
        {**dm_msg(4, [ALICE, BOT], timestamp=115), "flags": ["read", "wildcard_mentioned"]},
    ]

    tags = [line for line in search_messages(query="x").splitlines() if line.startswith("<msg ")]

    assert tags == [
        '<msg user="Alice" time="1970-01-01 00:01:40 UTC" stream="general" visibility="public" '
        'topic="t" id="1" unread="true" mentioned="true">',
        '<msg user="Alice" stream="general" visibility="public" topic="t" id="2">',
        f'<msg user="Alice" dm="{ALICE}, {BOT}" id="3">',
        f'<msg user="Alice" dm="{ALICE}, {BOT}" id="4" mentioned="true">',
    ]


def test_search_header_reports_after_bound(client: FakeClient) -> None:
    client.messages = [stream_msg(1, 100), stream_msg(2, 200)]

    out = search_messages(query="x", after="1970-01-01T00:02:00")

    assert out.startswith(f"Search as {BOT} (1 result; no older matches at or after 1970-01-01T00:02:00)")


def test_search_header_without_results(client: FakeClient) -> None:
    assert search_messages(query="x") == f"Search as {BOT} (0 results; no older matches)"


def test_search_header_reports_older_page(client: FakeClient) -> None:
    client.messages = [stream_msg(i) for i in range(1, 6)]

    out = search_messages(query="x", num_messages=2)

    assert out.startswith(f"Search as {BOT} (2 results; older: before_message_id=4)")


# --- user info and subscriptions ---------------------------------------------

@pytest.mark.parametrize("email, url", [
    (ALICE, f"/users/{ALICE}"),
    ("bob+tag@example.com", "/users/bob+tag@example.com"),
    ("../messages/999", "/users/..%2Fmessages%2F999"),
    ("x?y=1#z", "/users/x%3Fy%3D1%23z"),
])
def test_get_user_info_quotes_email(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch, email: str, url: str,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(client, "call_endpoint",
                        lambda **kwargs: calls.append(kwargs) or {"result": "success"},
                        raising=False)

    assert core.get_user_info(email) is None
    assert calls[0]["url"] == url


@pytest.mark.parametrize("allowlist, expected", [
    (None, ["general"]),
    ("secret", ["general", "secret"]),
    ("__ALL__", ["general", "ops", "secret"]),
])
def test_get_subscribed_streams_follows_allowlist(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
    allowlist: Optional[str], expected: list[str],
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)
    subs = [{"name": "secret", "invite_only": True}, {"name": "general", "invite_only": False},
            {"name": "ops", "invite_only": True}]
    monkeypatch.setattr(client, "get_subscriptions",
                        lambda: {"result": "success", "subscriptions": subs}, raising=False)

    assert [s["name"] for s in core.get_subscribed_streams()] == expected


def test_format_messages_dm_attribute() -> None:
    out = core.format_messages([dm_msg(1, [ALICE, BOT])], include_topic=True)

    assert f'dm="{ALICE}, {BOT}"' in out
    assert "stream=" not in out
    assert "dm=" not in core.format_messages([dm_msg(1, [ALICE, BOT])])



# --- read filters ------------------------------------------------------------

UNREAD = {"operator": "is", "operand": "unread"}
MENTIONED = {"operator": "is", "operand": "mentioned"}
IS_DM = {"operator": "is", "operand": "dm"}


def test_read_filters_are_enums_in_tool_schema() -> None:
    async def schemas() -> dict:
        return {t.name: t.parameters["properties"] for t in await mcp.mcp.list_tools()}

    props = asyncio.run(schemas())
    enum = {tool: props[tool]["only"]["anyOf"][0]["items"]["enum"]
            for tool in ("get_messages", "search_messages")}
    assert enum == {"get_messages": ["unread", "mentioned", "starred"],
                    "search_messages": ["unread", "mentioned", "dm", "starred"]}
    assert "dm_including" not in props["get_messages"]
    assert [name for name, p in props.items() if "dm_including" in p] == ["search_messages"]
    assert [name for name, p in props.items() if "max_chars" in p] == ["search_messages"]


def test_search_recipe_narrows(dm_reads: FakeClient) -> None:
    search_messages(only=["unread", "dm"])
    search_messages(only=["unread", "mentioned"])
    search_messages(query="x", dm_including=[ALICE, BOB], only=["unread"])

    assert [r["narrow"] for r in dm_reads.requests] == [
        [UNREAD, IS_DM],
        [UNREAD, MENTIONED],
        [{"operator": "dm-including", "operand": ALICE},
         {"operator": "dm-including", "operand": BOB},
         {"operator": "search", "operand": "x"}, UNREAD],
    ]


def test_search_non_dm_filter_keeps_default_scope(client: FakeClient) -> None:
    search_messages(only=["mentioned"])
    search_messages(only=["starred"], stream="general")

    assert [r["narrow"] for r in client.requests] == [
        PUBLIC + [MENTIONED],
        [{"operator": "channel", "operand": "general"}, {"operator": "is", "operand": "starred"}],
    ]


@pytest.mark.parametrize("allowlist", [None, "secret"])
@pytest.mark.parametrize("kwargs", [
    {"only": ["dm"]},
    {"only": ["unread", "dm"]},
    {"only": ["dm"], "scope": "received"},
    {"dm_including": [ALICE]},
    {"dm_including": [ALICE], "only": ["unread"]},
])
def test_search_dm_filters_denied_without_allow_all(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch,
    allowlist: Optional[str], kwargs: dict,
) -> None:
    if allowlist is not None:
        monkeypatch.setenv("BOT_ALLOWED_PRIVATE_STREAMS", allowlist)
    client.messages = [dm_msg(5, [ALICE, BOT], content="private hello")]

    assert search_messages(**kwargs) == DM_READS_ERROR
    assert client.requests == []


@pytest.mark.parametrize("narrow", [
    [IS_DM],
    [{"operator": "dm-including", "operand": ALICE}],
    [{"operator": "channel", "operand": "general"}, IS_DM],
    PUBLIC + [{"operator": "dm-including", "operand": ALICE}, UNREAD],
])
def test_gate_rejects_dm_filter_narrows(client: FakeClient, narrow: list[dict]) -> None:
    with pytest.raises(core.ReadScopeError) as exc:
        core.fetch_narrow(narrow)
    assert f"Error: {exc.value}" == DM_READS_ERROR
    assert client.requests == []


def test_gate_rejects_unscoped_flag_narrow(client: FakeClient) -> None:
    with pytest.raises(core.ReadScopeError, match=UNSCOPED):
        core.fetch_narrow([UNREAD])
    assert search_messages(only=["unread"], scope="received").startswith(f"Error: {UNSCOPED}")
    assert client.requests == []


def test_search_dm_filters_return_dms_with_allow_all(dm_reads: FakeClient) -> None:
    dm_reads.messages = [dm_msg(5, [ALICE, BOT], content="private hello")]

    assert "private hello" in search_messages(only=["unread", "dm"])
    assert "private hello" in search_messages(dm_including=[ALICE])


def test_get_messages_only_filters_conversation(dm_reads: FakeClient) -> None:
    dm_reads.messages = [stream_msg(1), dm_msg(2, [ALICE, BOT])]

    stream_out = get_messages(stream="general", topic="t", only=["unread", "mentioned"])
    dm_out = get_messages(dm_with=[ALICE], only=["unread"])
    get_messages(message_id=2, only=["starred"])

    assert [r["narrow"] for r in dm_reads.requests] == [
        [{"operator": "channel", "operand": "general"}, {"operator": "topic", "operand": "t"},
         UNREAD, MENTIONED],
        [{"operator": "dm", "operand": ALICE}, UNREAD],
        [{"operator": "dm", "operand": f"{ALICE},{BOT}"}, {"operator": "is", "operand": "starred"}],
    ]
    assert stream_out.startswith("Messages from [public] #general > t (only unread, mentioned):\n")
    assert dm_out.startswith(f"Messages from DM with {ALICE} (only unread):\n")


def test_get_messages_only_respects_read_gate(client: FakeClient) -> None:
    assert get_messages(dm_with=[ALICE], only=["unread"]) == DM_READS_ERROR
    assert get_messages(stream="secret", topic="t", only=["unread"]) == mcp.PRIVATE_STREAM_ERROR
    assert client.requests == []

    get_messages(stream="general", topic="t", only=["unread"])
    assert client.requests[0]["narrow"] == [
        {"operator": "channel", "operand": "general"}, {"operator": "topic", "operand": "t"}, UNREAD,
    ]


# --- search clipping ---------------------------------------------------------

def clipped(text: str, total: int) -> str:
    return f"{text}… [{total} chars; get_message_by_id has the rest]"


def hit_bodies(out: str) -> list[str]:
    """Content between each <msg ...> line and its </msg>."""
    body = out.split("\n\n", 1)[1]
    blocks = body.split("\n</msg>")
    assert blocks[-1] == ""
    return [block.lstrip("\n").split("\n", 1)[1] for block in blocks[:-1]]


@pytest.mark.parametrize("content, max_chars, expected", [
    ("hello", 10, "hello"),
    ("hello", 5, "hello"),
    ("hello world", 5, clipped("hello", 11)),
    ("🎉🎉🎉ok", 2, clipped("🎉🎉", 5)),
    ("hi", None, "hi"),
])
def test_search_max_chars(
    client: FakeClient, content: str, max_chars: Optional[int], expected: str,
) -> None:
    client.messages = [stream_msg(1, content=content)]

    assert hit_bodies(search_messages(query="x", max_chars=max_chars)) == [expected]


def test_search_max_chars_keeps_msg_wrappers(client: FakeClient) -> None:
    client.messages = [
        stream_msg(1, content="a <b>bold</b> line\nsecond line"),
        stream_msg(2, timestamp=1030, content="ok"),
    ]
    out = search_messages(query="x", max_chars=4)

    assert hit_bodies(out) == [clipped("a <b", 30), "ok"]
    tags = [line for line in out.splitlines() if line.startswith("<msg ")]
    assert len(tags) == out.count("\n</msg>") == 2
    assert out.endswith("\n</msg>")


@pytest.mark.parametrize("max_chars", [0, -5])
def test_search_rejects_invalid_max_chars(client: FakeClient, max_chars: int) -> None:
    assert search_messages(query="x", max_chars=max_chars) == "Error: max_chars must be at least 1."
    assert client.requests == []


def test_format_messages_does_not_combine_when_clipping(client: FakeClient) -> None:
    msgs = [stream_msg(1, content="first part"), stream_msg(2, timestamp=1060, content="second")]

    assert core.format_messages(msgs).count("<msg ") == 1
    out = core.format_messages(msgs, max_chars=5)

    assert out.count("<msg ") == 2
    assert f"\n{clipped('first', 10)}\n</msg>" in out
    assert f"\n{clipped('secon', 6)}\n</msg>" in out


def test_format_messages_hides_nobots_before_clipping(client: FakeClient) -> None:
    out = core.format_messages([stream_msg(1, content="/nobots secret"), stream_msg(2)], max_chars=3)

    assert "secret" not in out and "/no" not in out
    assert 'id="2"' in out
