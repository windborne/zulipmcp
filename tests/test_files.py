import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zulipmcp import core

mcp = importlib.import_module("zulipmcp.mcp")
fetch_file = getattr(mcp.fetch_file, "fn", mcp.fetch_file)
fetch_image = getattr(mcp.fetch_image, "fn", mcp.fetch_image)

SITE = "https://zulip.example.com"


class FakeSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> SimpleNamespace:
        self.urls.append(url)
        return SimpleNamespace(status_code=200, content=b"payload",
                               headers={"Content-Type": "image/png"}, text="")


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    fake = FakeSession()
    client = SimpleNamespace(base_url=f"{SITE}/api/", session=fake, ensure_session=lambda: None)
    monkeypatch.setattr(core, "get_client", lambda: client)
    return fake


@pytest.mark.parametrize("path", [
    "/api/v1/messages?narrow=[]",
    "/api/v1/users/me",
    "/user_uploads/2/ab/payload.txt?/.bashrc",
    "/user_uploads/2/ab/payload.txt#frag",
    "/user_uploads/../api/v1/messages",
    "/user_uploads/2/%2e%2e/%2E%2E/api/v1/messages",
    "https://other.example.com/user_uploads/2/ab/file.txt",
    "//other.example.com/user_uploads/2/ab/file.txt",
    "user_uploads/2/ab/file.txt",
    "/static/user_uploads/file.txt",
])
def test_rejects_non_upload_paths(session: FakeSession, tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="Only /user_uploads/"):
        core.save_file(path, save_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Only /user_uploads/"):
        core.save_image(path)
    assert session.urls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["/user_uploads/2/ab/", "/user_uploads/2/ab/.bashrc"])
def test_rejects_unsafe_filenames(session: FakeSession, tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="safe filename"):
        core.save_file(path, save_dir=str(tmp_path))
    assert session.urls == []


def test_saves_basename_into_save_dir(session: FakeSession, tmp_path: Path) -> None:
    saved, size, content_type = core.save_file("/user_uploads/2/ab/report.pdf",
                                               save_dir=str(tmp_path / "out"))

    assert Path(saved) == tmp_path / "out" / "report.pdf"
    assert Path(saved).read_bytes() == b"payload"
    assert (size, content_type) == (7, "image/png")
    assert session.urls == [f"{SITE}/user_uploads/2/ab/report.pdf"]


def test_saves_into_fresh_temp_dir_by_default(session: FakeSession) -> None:
    saved, _, _ = core.save_file("/user_uploads/2/ab/report.pdf")

    assert Path(saved).name == "report.pdf"
    assert Path(saved).read_bytes() == b"payload"
    shutil.rmtree(Path(saved).parent)


def test_refuses_to_overwrite(session: FakeSession, tmp_path: Path) -> None:
    existing = tmp_path / "report.pdf"
    existing.write_bytes(b"original")

    out = fetch_file("/user_uploads/2/ab/report.pdf", save_dir=str(tmp_path))

    assert out == f"Error fetching file: File already exists: {existing}"
    assert existing.read_bytes() == b"original"
    assert session.urls == []


def test_tools_report_rejected_paths(session: FakeSession) -> None:
    assert fetch_file("/api/v1/messages").startswith("Error fetching file: Only /user_uploads/")
    assert fetch_image("/api/v1/messages").startswith("Error fetching image: Only /user_uploads/")
    assert session.urls == []


def test_fetch_image_saves_upload(session: FakeSession) -> None:
    out = fetch_image("/user_uploads/2/ab/pic.png")

    path = Path(out.split("\n")[0].removeprefix("Image saved to: "))
    assert path.suffix == ".png"
    assert path.read_bytes() == b"payload"
    path.unlink()
