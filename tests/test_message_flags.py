"""Tests for the update_message_flags tool."""
import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from zulipmcp import core
from zulipmcp.mcp import mcp as server


class _FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    def update_message_flags(self, request: dict) -> dict:
        self.requests.append(request)
        return self.response


def _call(arguments: dict) -> str:
    async def run() -> str:
        async with Client(server) as client:
            result = await client.call_tool("update_message_flags", arguments)
            return result.content[0].text

    return asyncio.run(run())


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient({"result": "success", "msg": "", "messages": [1, 2]})
    monkeypatch.setattr(core, "get_client", lambda: client)
    return client


def test_star_messages(fake: _FakeClient) -> None:
    out = _call({"message_ids": [1, 2], "flag": "starred", "op": "add"})

    assert fake.requests == [{"messages": [1, 2], "op": "add", "flag": "starred"}]
    assert out == "Added flag 'starred' on 2 message(s): [1, 2]"


def test_mark_unread_reports_skipped_channels(fake: _FakeClient) -> None:
    fake.response = {"result": "success", "msg": "", "messages": [1],
                     "ignored_because_not_subscribed_channels": [7]}

    out = _call({"message_ids": [1, 2], "flag": "read", "op": "remove"})

    assert fake.requests[0]["op"] == "remove"
    assert out.startswith("Removed flag 'read' on 1 message(s): [1]")
    assert "unsubscribed channel IDs: [7]" in out


@pytest.mark.parametrize("arguments", [
    {"message_ids": [1], "flag": "starred", "op": "add "},
    {"message_ids": [1], "flag": "starred", "op": "toggle"},
    {"message_ids": [1], "flag": "mentioned", "op": "add"},
])
def test_rejects_invalid_op_or_flag_before_calling_zulip(
    fake: _FakeClient, arguments: dict,
) -> None:
    with pytest.raises(ToolError):
        _call(arguments)

    assert fake.requests == []


def test_rejects_empty_message_list(fake: _FakeClient) -> None:
    out = _call({"message_ids": [], "flag": "starred", "op": "add"})

    assert out.startswith("Error:")
    assert fake.requests == []


def test_reports_api_error(fake: _FakeClient) -> None:
    fake.response = {"result": "error", "msg": "Invalid message(s)"}

    out = _call({"message_ids": [99], "flag": "starred", "op": "add"})

    assert out == "Error updating message flags: Invalid message(s)"
