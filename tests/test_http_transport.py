"""Tests for the Streamable HTTP transport and its optional bearer-token auth."""
import importlib
from collections.abc import Iterator

import pytest
from fastmcp.server.auth import AccessToken, AuthProvider
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from starlette.testclient import TestClient

from zulipmcp import __main__ as cli
from zulipmcp.mcp import _auth_from_env, mcp

zmcp = importlib.import_module("zulipmcp.mcp")  # the package re-exports the FastMCP object as `mcp`

BASE_URL = "https://mcp.example.com"
ISSUER = "https://auth.example.com"
KEY = RSAKeyPair.generate()
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}},
}
HEADERS = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZULIPMCP_AUTH_JWKS_URI", f"{ISSUER}/jwks.json")
    monkeypatch.setenv("ZULIPMCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("ZULIPMCP_BASE_URL", BASE_URL)
    monkeypatch.delenv("ZULIPMCP_AUTH_AUDIENCE", raising=False)
    monkeypatch.delenv("ZULIPMCP_AUTH_REQUIRED_SCOPES", raising=False)


def _client(transport: str, path: str, stateless: bool = False) -> Iterator[TestClient]:
    auth = _auth_from_env(path)
    if auth is not None:
        auth.token_verifier.public_key = KEY.public_key  # skip the JWKS fetch
    mcp.auth = auth
    try:
        app = mcp.http_app(transport=transport, stateless_http=stateless)
        with TestClient(app, base_url=BASE_URL) as client:
            yield client
    finally:
        mcp.auth = None


@pytest.fixture
def http_client(auth_env: None) -> Iterator[TestClient]:
    yield from _client("http", "/mcp", stateless=True)


def _token(audience: str) -> str:
    return KEY.create_token(issuer=ISSUER, audience=audience)


@pytest.fixture(autouse=True)
def no_log_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_server() would set up a real log file; tests/test_logging.py covers it."""
    monkeypatch.setattr(zmcp, "_setup_logging", lambda: None)


@pytest.fixture
def clean_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_server() reads these; keep the caller's shell out of the test."""
    for name in ("ZULIPMCP_READ_ONLY", "SESSION_STREAM", "SESSION_TOPIC"):
        monkeypatch.delenv(name, raising=False)


AUTH_VARS = ("ZULIPMCP_AUTH_JWKS_URI", "ZULIPMCP_AUTH_ISSUER", "ZULIPMCP_AUTH_AUDIENCE",
             "ZULIPMCP_AUTH_REQUIRED_SCOPES", "ZULIPMCP_BASE_URL")


@pytest.fixture
def no_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in AUTH_VARS:
        monkeypatch.delenv(name, raising=False)


def test_auth_disabled_without_any_auth_variable(no_auth_env: None) -> None:
    assert _auth_from_env("/mcp") is None


@pytest.mark.parametrize("name", ["ZULIPMCP_AUTH_ISSUER", "ZULIPMCP_BASE_URL",
                                  "ZULIPMCP_AUTH_AUDIENCE", "ZULIPMCP_AUTH_REQUIRED_SCOPES",
                                  "ZULIPMCP_AUTH_JWKS_URL"])
def test_auth_variable_without_jwks_fails_closed(
    no_auth_env: None, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.setenv(name, "https://auth.example.com")
    with pytest.raises(ValueError, match=name):
        _auth_from_env("/mcp")


def test_partial_auth_config_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZULIPMCP_AUTH_JWKS_URI", f"{ISSUER}/jwks.json")
    monkeypatch.delenv("ZULIPMCP_AUTH_ISSUER", raising=False)
    with pytest.raises(ValueError):
        _auth_from_env("/mcp")


@pytest.mark.parametrize("transport,path", [("http", "/mcp"), ("sse", "/sse")])
def test_challenge_points_at_metadata_for_this_path(
    auth_env: None, transport: str, path: str,
) -> None:
    for client in _client(transport, path):
        if transport == "http":
            response = client.post(path, json=INIT, headers=HEADERS)
        else:
            response = client.get(path)
        assert response.status_code == 401
        challenge = response.headers["www-authenticate"]
        metadata_url = challenge.split('resource_metadata="')[1].split('"')[0]
        assert metadata_url == f"{BASE_URL}/.well-known/oauth-protected-resource{path}"

        metadata = client.get(metadata_url).json()
        assert metadata["resource"] == BASE_URL + path
        assert metadata["authorization_servers"] == [ISSUER]


def test_token_for_other_resource_is_rejected_at_http_boundary(
    http_client: TestClient,
) -> None:
    token = _token(BASE_URL + "/sse")
    response = http_client.post("/mcp", json=INIT,
                                headers={**HEADERS, "Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_stateless_initialize_with_valid_token(http_client: TestClient) -> None:
    token = _token(BASE_URL + "/mcp")
    response = http_client.post("/mcp", json=INIT,
                                headers={**HEADERS, "Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "mcp-session-id" not in response.headers
    assert '"serverInfo"' in response.text


def test_stateful_http_issues_session_id(no_auth_env: None) -> None:
    for client in _client("http", "/mcp"):
        response = client.post("/mcp", json=INIT, headers=HEADERS)
        assert response.status_code == 200
        assert response.headers.get("mcp-session-id")


def test_run_server_rejects_bad_auth_config_before_zulip_calls(
    no_auth_env: None, clean_server_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZULIPMCP_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("SESSION_STREAM", "general")
    monkeypatch.setenv("SESSION_TOPIC", "t")
    calls: list[str] = []
    monkeypatch.setattr(zmcp, "_init_session", lambda *a, **k: calls.append("init") or "ok")
    monkeypatch.setattr(mcp, "run", lambda **kwargs: calls.append("run"))

    with pytest.raises(ValueError, match="ZULIPMCP_AUTH_ISSUER"):
        zmcp.run_server(transport="http")
    assert calls == []


class _StubAuth(AuthProvider):
    async def verify_token(self, token: str) -> AccessToken | None:
        if token != "good":
            return None
        return AccessToken(token=token, client_id="stub", scopes=[])


def test_run_server_uses_caller_auth_provider(
    no_auth_env: None, clean_server_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZULIPMCP_AUTH_ISSUER", ISSUER)  # ignored when auth is passed
    ran: list[dict] = []
    monkeypatch.setattr(mcp, "run", lambda **kwargs: ran.append(kwargs))
    stub = _StubAuth()
    try:
        zmcp.run_server(transport="http", stateless=True, auth=stub)
        assert mcp.auth is stub
        assert ran == [{"transport": "http", "host": "127.0.0.1", "port": 8235,
                        "stateless_http": True}]
        with TestClient(mcp.http_app(transport="http", stateless_http=True)) as client:
            for token, status in (("good", 200), ("bad", 401)):
                response = client.post("/mcp", json=INIT, headers={
                    **HEADERS, "Authorization": f"Bearer {token}"})
                assert response.status_code == status
    finally:
        mcp.auth = None


def test_caller_auth_rejected_on_stdio(clean_server_env: None) -> None:
    with pytest.raises(ValueError, match="sse and http"):
        zmcp.run_server(transport="stdio", auth=_StubAuth())


@pytest.mark.parametrize("transport", ["stdio", "sse"])
def test_stateless_requires_http_transport(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], transport: str,
) -> None:
    monkeypatch.setattr(cli, "run_server", lambda **kwargs: pytest.fail("should not start"))
    monkeypatch.setattr("sys.argv", ["zulipmcp", "--transport", transport, "--stateless"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "--stateless requires --transport http" in capsys.readouterr().err


@pytest.mark.parametrize("transport", ["stdio", "sse"])
def test_run_server_rejects_stateless_without_http(
    clean_server_env: None, monkeypatch: pytest.MonkeyPatch, transport: str,
) -> None:
    monkeypatch.setattr(mcp, "run", lambda **kwargs: pytest.fail("should not start"))
    with pytest.raises(ValueError, match="stateless"):
        zmcp.run_server(transport=transport, stateless=True)
