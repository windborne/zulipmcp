"""Tests for the per-request Zulip client resolver (core.set_client_resolver)."""
import asyncio
import importlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
from fastmcp.server.auth import AccessToken, AuthProvider
from fastmcp.server.dependencies import get_access_token
from starlette.testclient import TestClient

from zulipmcp import core
from zulipmcp.mcp import mcp

zmcp = importlib.import_module("zulipmcp.mcp")  # the package re-exports the FastMCP object as `mcp`

SESSION_TOOLS = {"set_context", "reply", "listen", "end_session", "typing", "stop_typing"}
LOCAL_FS_TOOLS = {"fetch_image", "fetch_file", "upload_file"}
HEADERS = {"Accept": "application/json, text/event-stream", "Authorization": "Bearer alice"}


class _StubAuth(AuthProvider):
    async def verify_token(self, token: str) -> AccessToken | None:
        return AccessToken(token=token, client_id="stub", scopes=[])


class _FakeClient:
    email = "alice@example.com"
    base_url = "https://zulip.example.com/api/"

    def get_subscriptions(self) -> dict:
        return {"result": "success", "subscriptions": []}


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[str]]:
    """A real zuliprc the resolver must ignore, no live HTTP, and a server left as found."""
    monkeypatch.setattr(zmcp, "_setup_logging", lambda: None)
    for name in ("ZULIPMCP_READ_ONLY", "SESSION_STREAM", "SESSION_TOPIC", "ZULIPMCP_AUTH_JWKS_URI",
                 "ZULIPMCP_AUTH_ISSUER", "ZULIPMCP_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    rc = tmp_path / "zuliprc"
    rc.write_text("[api]\nemail=bot@example.com\nkey=k\nsite=https://zulip.example.com\n")
    monkeypatch.setenv("ZULIP_RC_PATH", str(rc))
    monkeypatch.setattr(core, "_client", None)
    sent: list[str] = []
    monkeypatch.setattr(requests.Session, "send", lambda self, req, **kw: sent.append(req.url))
    monkeypatch.setattr(mcp, "_transforms", list(mcp._transforms))  # undo run_server's disable()
    monkeypatch.setattr(mcp, "run", lambda **kwargs: None)
    yield sent
    core.set_client_resolver(None)
    mcp.auth = None


def _serve(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SESSION_STREAM", "general")
    monkeypatch.setenv("SESSION_TOPIC", "t")
    monkeypatch.setattr(zmcp, "_init_session", lambda *a, **k: pytest.fail("auto-init in resolver mode"))
    zmcp.run_server(transport="http", stateless=True, auth=_StubAuth())
    return TestClient(mcp.http_app(transport="http", stateless_http=True))


def _rpc(client: TestClient, method: str, params: dict) -> dict:
    response = client.post("/mcp", headers=HEADERS,
                           json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    assert response.status_code == 200
    data = next(line for line in response.text.splitlines() if line.startswith("data: "))
    return json.loads(data[len("data: "):])["result"]


def test_get_client_uses_resolver_not_zuliprc() -> None:
    fake = _FakeClient()
    core.set_client_resolver(lambda: fake)
    assert core.get_client() is fake
    assert core._client is None
    core.set_client_resolver(None)
    assert not core.has_client_resolver()


def test_tool_runs_against_the_requests_client(
    monkeypatch: pytest.MonkeyPatch, isolated: list[str],
) -> None:
    seen: list[str] = []

    def resolver() -> _FakeClient:
        token = get_access_token()
        assert token is not None
        seen.append(token.token)
        return _FakeClient()

    core.set_client_resolver(resolver)
    with _serve(monkeypatch) as client:
        result = _rpc(client, "tools/call", {"name": "get_subscribed_streams", "arguments": {}})
    assert not result["isError"]
    assert result["content"][0]["text"] == "Not subscribed to any streams."
    assert seen == ["alice"]
    assert isolated == [] and core._client is None


def test_resolver_error_fails_closed(monkeypatch: pytest.MonkeyPatch, isolated: list[str]) -> None:
    def resolver() -> _FakeClient:
        raise PermissionError("no grant for this token")

    core.set_client_resolver(resolver)
    with _serve(monkeypatch) as client:
        result = _rpc(client, "tools/call", {"name": "get_subscribed_streams", "arguments": {}})
    assert result["isError"]
    assert "no grant for this token" in result["content"][0]["text"]
    assert isolated == [] and core._client is None


def test_resolver_mode_hides_session_and_local_fs_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    core.set_client_resolver(_FakeClient)
    with _serve(monkeypatch) as client:
        names = {tool["name"] for tool in _rpc(client, "tools/list", {})["tools"]}
    assert names & (SESSION_TOOLS | LOCAL_FS_TOOLS) == set()
    assert {"get_subscribed_streams", "send_message"} <= names


@pytest.mark.parametrize("tag,tools", [("SESSION_TAG", SESSION_TOOLS), ("LOCAL_FS_TAG", LOCAL_FS_TOOLS)])
def test_tag_marks_exactly_these_tools(tag: str, tools: set[str]) -> None:
    tagged = {t.name for t in asyncio.run(mcp.list_tools()) if getattr(zmcp, tag) in t.tags}
    assert tagged == tools


def test_human_user_id_per_account(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Human(_FakeClient):
        def __init__(self, user_id: int) -> None:
            self.user_id = user_id

        def get_profile(self) -> dict:
            return {"result": "success", "is_bot": False, "user_id": self.user_id}

    alice, bob = _Human(7), _Human(8)
    monkeypatch.setattr(core, "_account", None)
    for client, expected in ((alice, 7), (bob, 8), (alice, 7)):
        core.set_client_resolver(lambda c=client: c)
        assert core._human_user_id() == expected

    class _SwappedMidCall(tuple):
        """Stands in for another account's call replacing the cache right after the check."""
        def __getitem__(self, i: int):  # type: ignore[override]
            if i == 0:
                core._account = (bob, 8)
            return tuple.__getitem__(self, i)

    monkeypatch.setattr(core, "_account", _SwappedMidCall((alice, 7)))
    core.set_client_resolver(lambda: alice)
    assert core._human_user_id() == 7


def test_resolver_refused_in_read_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZULIPMCP_READ_ONLY", "1")
    core.set_client_resolver(lambda: pytest.fail("resolver called in read-only mode"))
    with pytest.raises(core.ReadOnlyError):
        core.get_client()
