from pathlib import Path
from types import SimpleNamespace

import pytest

from zulipmcp import core


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ZULIPMCP_CACHE_DIR", str(tmp_path))
    return tmp_path


def _use_identity(monkeypatch: pytest.MonkeyPatch, email: str,
                  site: str = "https://zulip.example.com") -> None:
    client = SimpleNamespace(email=email, base_url=f"{site}/api/")
    monkeypatch.setattr(core, "get_client", lambda: client)


def test_cache_is_scoped_per_identity(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_identity(monkeypatch, "alice@example.com")
    alice = core._get_cache()
    alice.set("key", "alice-value")

    _use_identity(monkeypatch, "alice-bot@example.com")
    bot = core._get_cache()

    assert Path(alice.directory) != Path(bot.directory)
    assert Path(alice.directory).parent == cache_dir
    assert bot.get("key") is None


def test_same_email_on_different_sites_is_isolated(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_identity(monkeypatch, "alice@example.com", "https://one.example.com")
    one = core._get_cache().directory
    _use_identity(monkeypatch, "alice@example.com", "https://two.example.com")
    two = core._get_cache().directory

    assert one != two


def test_same_identity_reuses_cache(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_identity(monkeypatch, "alice@example.com")

    assert core._get_cache() is core._get_cache()
