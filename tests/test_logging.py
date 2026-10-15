"""Logging is the server's business: importing zulipmcp configures nothing."""
import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

mcp = importlib.import_module("zulipmcp.mcp")


def test_import_configures_no_logging_and_writes_no_files(tmp_path: Path) -> None:
    code = (
        "import logging, zulipmcp, zulipmcp.core\n"
        "assert not logging.getLogger().handlers, logging.getLogger().handlers\n"
        "assert not logging.getLogger('zulipmcp').handlers, logging.getLogger('zulipmcp').handlers\n"
    )
    env = {**os.environ, "ZULIPMCP_LOG_DIR": str(tmp_path / "logs")}

    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path, env=env)

    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def fresh_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A process that hasn't set up logging yet; undone afterwards."""
    monkeypatch.setenv("ZULIPMCP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mcp, "_logging_ready", False)
    monkeypatch.setattr(mcp._logger, "handlers", [])
    root = logging.getLogger()
    root_handlers, level = list(root.handlers), mcp._logger.level
    yield
    for handler in mcp._logger.handlers:
        handler.close()
    for handler in [h for h in root.handlers if h not in root_handlers]:
        root.removeHandler(handler)
    mcp._logger.setLevel(level)


def _file_handlers() -> list[logging.Handler]:
    return [h for h in mcp._logger.handlers if isinstance(h, logging.FileHandler)]


def test_setup_logging_twice_adds_one_file_handler(fresh_logging: None, tmp_path: Path) -> None:
    mcp._setup_logging()
    mcp._setup_logging()

    assert len(_file_handlers()) == 1
    assert len(list(tmp_path.glob("mcp_*.log"))) == 1


def test_setup_logging_ignores_handlers_a_host_attached(fresh_logging: None) -> None:
    mcp._logger.addHandler(logging.NullHandler())

    mcp._setup_logging()

    assert len(_file_handlers()) == 1
