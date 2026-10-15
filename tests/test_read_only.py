"""Tests for ZULIPMCP_READ_ONLY.

Contract: with read-only on, the client refuses every non-GET request before
it reaches the transport, and the MCP server hides every write-tagged tool.
"""
import asyncio
import importlib
import json
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import pytest
import requests
import zulip
from requests.adapters import HTTPAdapter
from fastmcp import FastMCP

from zulipmcp import core

# The package re-exports its FastMCP instance as `zulipmcp.mcp`, which
# shadows the submodule for `import zulipmcp.mcp as ...` — load it by name.
mcp = importlib.import_module("zulipmcp.mcp")

READ_TOOLS = {
    "fetch_file",
    "fetch_image",
    "get_message_by_id",
    "get_message_link",
    "get_messages",
    "get_stream_members",
    "get_stream_topics",
    "get_subscribed_streams",
    "get_user_info",
    "list_emoji",
    "list_streams",
    "resolve_name",
    "search_messages",
    "verify_message",
}

WRITE_TOOLS = {
    "add_reaction",
    "edit_message",
    "end_session",
    "listen",
    "move_messages",
    "remove_reaction",
    "reply",
    "resolve_topic",
    "send_direct_message",
    "send_message",
    "set_context",
    "stop_typing",
    "typing",
    "upload_file",
}

# Canned bodies for the GETs the write paths make before their write.
_GET_ROUTES: dict[str, dict[str, Any]] = {
    "server_settings": {"zulip_version": "12.0", "zulip_feature_level": 500},
    "streams": {"streams": [{"name": "general", "invite_only": False}]},
    "get_stream_id": {"stream_id": 7},
    "users/me/subscriptions": {"subscriptions": []},
    "users/me": {"user_id": 3, "is_bot": False},
    "messages": {"messages": [
        {"id": 1, "type": "stream", "display_recipient": "general", "subject": "t"},
    ]},
    "messages/1": {"message": {"id": 1, "type": "stream", "display_recipient": "general", "subject": "t"}},
}


@pytest.fixture
def transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Answer known GETs offline; fail loudly on any non-GET. Returns the request log."""
    requests_seen: list[tuple[str, str]] = []

    def fake_send(self: HTTPAdapter, request: requests.PreparedRequest,
                  *args: Any, **kwargs: Any) -> requests.Response:
        method, url = request.method or "", request.url or ""
        path = urllib.parse.urlparse(url).path.replace("//", "/")
        route = path.removeprefix("/api/v1/")
        requests_seen.append((method, route))
        if method != "GET":
            raise AssertionError(f"non-GET reached the transport: {method} {url}")
        if route not in _GET_ROUTES:
            raise AssertionError(f"unexpected GET: {url}")
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response.request = request
        response._content = json.dumps({"result": "success", "msg": "", **_GET_ROUTES[route]}).encode()
        return response

    rc = tmp_path / "zuliprc"
    rc.write_text("[api]\nemail=bot@example.com\nkey=test-key\nsite=https://zulip.example.com\n")
    monkeypatch.setenv("ZULIP_RC_PATH", str(rc))
    monkeypatch.delenv("BOT_ALLOWED_WRITE_STREAMS", raising=False)
    monkeypatch.delenv("BOT_ALLOWED_PRIVATE_STREAMS", raising=False)
    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    monkeypatch.setattr(core, "_client", None)
    monkeypatch.setenv("ZULIPMCP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(core, "_stream_id_cache", {})
    return requests_seen


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On", " true "])
def test_read_only_enabled_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", value)

    assert core.is_read_only()


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "No", "OFF"])
def test_read_only_disabled_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", value)

    assert not core.is_read_only()


def test_read_only_unset_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZULIPMCP_READ_ONLY", raising=False)

    assert not core.is_read_only()


@pytest.mark.parametrize("value", ["2", "ture", "enabled", "1 true"])
def test_read_only_unrecognized_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", value)

    with pytest.raises(ValueError, match="Invalid ZULIPMCP_READ_ONLY"):
        core.is_read_only()


def test_run_server_rejects_unrecognized_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "ture")
    ran: list[dict] = []
    monkeypatch.setattr(mcp.mcp, "run", lambda **kwargs: ran.append(kwargs))

    with pytest.raises(ValueError, match="Invalid ZULIPMCP_READ_ONLY"):
        mcp.run_server()
    assert ran == []


def _upload(tmp_path: Path) -> object:
    path = tmp_path / "upload.txt"
    path.write_text("hello")
    return core.upload_file(str(path))


_WRITES: dict[str, Callable[[Path], object]] = {
    "send_message": lambda _: core.send_message("general", "t", "hi"),
    "send_direct_message": lambda _: core.send_direct_message(["user@example.com"], "hi"),
    "add_reaction": lambda _: core.add_reaction(1, "thumbs_up"),
    "edit_message": lambda _: core.edit_message(1, "edited"),
    "send_typing": lambda _: core.send_typing("general", "t", "start"),
    "upload_file": _upload,
    "ensure_subscribed": lambda _: core.ensure_subscribed("general"),
}


@pytest.mark.parametrize("write", _WRITES.values(), ids=_WRITES.keys())
def test_read_only_blocks_writes_before_network(
    transport: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write: Callable[[Path], object],
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")

    with pytest.raises(core.ReadOnlyError, match=r"refusing (POST|PATCH) https://zulip\.example\.com/api/v1/"):
        write(tmp_path)

    assert transport
    assert all(method == "GET" for method, _ in transport)


def test_read_only_get_reaches_transport(
    transport: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")
    client = core.get_client()
    assert isinstance(client, core._ReadOnlyClient)

    assert client.call_endpoint(url="messages", method="GET")["result"] == "success"
    assert client.call_endpoint(url="messages", method="get")["result"] == "success"
    assert transport[-2:] == [("GET", "messages"), ("GET", "messages")]

    with pytest.raises(core.ReadOnlyError):
        client.call_endpoint(url="messages", method="post")
    assert all(method == "GET" for method, _ in transport)


def test_read_only_unread_marker_fetches_profile_with_get(
    transport: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")
    msg = {"id": 1, "timestamp": 0, "type": "stream", "content": "hi", "flags": []}

    assert 'unread="true"' in core.format_messages([msg])
    assert transport[-1] == ("GET", "users/me")
    assert all(method == "GET" for method, _ in transport)


@pytest.mark.parametrize("verb", ["post", "put", "patch", "delete", "head"])
def test_read_only_direct_session_use_is_blocked(
    transport: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")
    client = core.get_client()
    client.ensure_session()
    assert client.session is not None

    with pytest.raises(core.ReadOnlyError, match=f"refusing {verb.upper()} "):
        getattr(client.session, verb)("https://zulip.example.com/api/v1/messages")
    assert all(method == "GET" for method, _ in transport)


def test_read_only_fails_closed_without_guarded_session(
    transport: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")
    monkeypatch.setattr(core._ReadOnlyClient, "ensure_session", zulip.Client.ensure_session)

    with pytest.raises(RuntimeError, match="session is unguarded"):
        core.get_client()
    assert core._client is None


def test_read_only_off_uses_plain_client(
    transport: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZULIPMCP_READ_ONLY", raising=False)

    assert type(core.get_client()) is zulip.Client


def _tool_names(read_only: bool) -> set[str]:
    """List tools on a copy of the server so the module-level instance is untouched."""
    async def names() -> set[str]:
        server = FastMCP("test")
        for tool in await mcp.mcp.list_tools():
            server.add_tool(tool)
        if read_only:
            server.disable(tags={mcp.WRITE_TAG})
        return {tool.name for tool in await server.list_tools()}

    return asyncio.run(names())


def test_read_only_exposes_only_read_tools() -> None:
    assert _tool_names(read_only=True) == READ_TOOLS


def test_all_tools_present_without_read_only() -> None:
    assert _tool_names(read_only=False) == READ_TOOLS | WRITE_TOOLS


@pytest.mark.parametrize("read_only", [True, False])
def test_run_server_read_only_wiring(monkeypatch: pytest.MonkeyPatch, read_only: bool) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1" if read_only else "0")
    monkeypatch.setenv("SESSION_STREAM", "general")
    monkeypatch.setenv("SESSION_TOPIC", "t")
    disabled: list[dict] = []
    inits: list[tuple[str, str]] = []
    monkeypatch.setattr(mcp.mcp, "disable", lambda **kwargs: disabled.append(kwargs))
    monkeypatch.setattr(mcp, "_init_session", lambda s, t, num_messages=0: inits.append((s, t)) or "ok")
    monkeypatch.setattr(mcp.mcp, "run", lambda **kwargs: None)

    mcp.run_server()

    assert disabled == ([{"tags": {mcp.WRITE_TAG}}] if read_only else [])
    assert inits == ([] if read_only else [("general", "t")])
