import json

from zulipmcp.listener import _session_error


def _write(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_api_error_result_is_reported(tmp_path):
    log = tmp_path / "s.jsonl"
    _write(log, [
        {"type": "system", "subtype": "init"},
        {"type": "result", "is_error": True, "terminal_reason": "api_error",
         "result": "Failed to authenticate: OAuth session expired and could not be refreshed"},
    ])
    assert "OAuth session expired" in _session_error(log)


def test_clean_result_is_not_reported(tmp_path):
    log = tmp_path / "s.jsonl"
    _write(log, [{"type": "result", "is_error": False, "terminal_reason": "completed",
                  "result": "done"}])
    assert _session_error(log) is None


def test_missing_or_garbage_log_is_not_reported(tmp_path):
    assert _session_error(tmp_path / "missing.jsonl") is None
    log = tmp_path / "s.jsonl"
    log.write_text("not json\n[1, 2]\n")
    assert _session_error(log) is None
