"""zulipmcp.listener — @mention → Claude Code session spawner.

Watches Zulip for @mentions via long-polling and spawns one headless Claude
Code subprocess per (stream, topic). Pairs with the zulipmcp MCP server —
Claude gets Zulip tools via MCP, the listener just handles lifecycle.

Usage:
    python -m zulipmcp.listener --zuliprc .zuliprc
    python -m zulipmcp.listener --zuliprc .zuliprc --mcp-config .mcp.json --system-prompt agent.md

What it does:
    1. Long-polls Zulip for message events
    2. Filters for stream @mentions not from self
    3. Deduplicates by (stream, topic) — one session at a time
    4. Spawns `claude` with --mcp-config pointing at zulipmcp
    5. Sets TRIGGER_MESSAGE_ID so set_context() anchors correctly
    6. Reaps finished processes via daemon monitor threads

What it deliberately omits (add when you need them):
    - Concurrency cap / queue
    - Workspace isolation
    - Staleness watchdog
    - Session history database
    - Dashboard / HTTP server
    - Systemd integration
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import zulip


@dataclass
class Config:
    """CLI args map 1:1 to fields."""
    zuliprc: str
    mcp_config: str = ""
    system_prompt: str = ""
    working_dir: str = "."
    claude_command: str = "claude"
    log_dir: str = "./logs"


# One session per (stream, topic). Value is the Popen object.
_sessions: dict[tuple[str, str], subprocess.Popen] = {}
_lock = threading.Lock()


def _build_cmd(cfg: Config, stream: str, topic: str) -> list[str]:
    """Assemble the claude CLI invocation."""
    cmd = [cfg.claude_command, "--dangerously-skip-permissions",
           "--output-format", "stream-json", "--verbose"]
    if cfg.mcp_config and Path(cfg.mcp_config).exists():
        cmd += ["--mcp-config", str(Path(cfg.mcp_config).resolve())]
    if cfg.system_prompt and Path(cfg.system_prompt).exists():
        cmd += ["--append-system-prompt", Path(cfg.system_prompt).read_text()]
    cmd += ["-p", f"Call set_context('{stream}', '{topic}') to begin, "
                   f"then greet the user and listen for follow-ups."]
    return cmd


def _build_env(cfg: Config, msg: dict) -> dict:
    """Set env vars that zulipmcp reads on the other side."""
    env = os.environ.copy()
    env["ZULIP_RC_PATH"] = str(Path(cfg.zuliprc).resolve())
    env["CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"] = "10800000"  # 3 hours
    env["TRIGGER_MESSAGE_ID"] = str(msg["id"])
    env["SESSION_USER_EMAIL"] = msg.get("sender_email", "")
    return env


def _spawn(cfg: Config, msg: dict):
    """Launch a Claude Code session for this message, if not already running."""
    stream, topic = msg["display_recipient"], msg["subject"]
    key = (stream.lower(), topic.lower())

    with _lock:
        p = _sessions.get(key)
        if p and p.poll() is None:
            return  # session alive

    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(cfg.log_dir) / f"{ts}_{key[0]}_{key[1]}.jsonl"
    fh = open(log_path, "w")

    proc = subprocess.Popen(
        _build_cmd(cfg, stream, topic),
        cwd=cfg.working_dir, env=_build_env(cfg, msg),
        stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    with _lock:
        _sessions[key] = proc

    def reap():
        proc.wait()
        fh.close()
        with _lock:
            _sessions.pop(key, None)

    threading.Thread(target=reap, daemon=True, name=f"reap:{key[0]}/{key[1]}").start()
    print(f"[listener] Spawned session for #{stream} > {topic} (pid={proc.pid})", file=sys.stderr)


def run(cfg: Config):
    """Block forever, dispatching @mentions to _spawn()."""
    client = zulip.Client(config_file=cfg.zuliprc)
    me = client.get_profile()["user_id"]
    print(f"[listener] Listening as user_id={me}, log_dir={cfg.log_dir}", file=sys.stderr)

    def handle(event):
        msg = event["message"]
        if (msg.get("type") == "stream"
                and "mentioned" in event.get("flags", [])
                and msg.get("sender_id") != me):
            _spawn(cfg, msg)

    client.call_on_each_event(handle, event_types=["message"])


def main():
    p = argparse.ArgumentParser(
        description="Zulip → Claude Code listener",
        epilog="See zulipmcp README for full setup instructions.",
    )
    p.add_argument("--zuliprc", required=True, help="Path to .zuliprc")
    p.add_argument("--mcp-config", default="", help="Path to .mcp.json for Claude")
    p.add_argument("--system-prompt", default="", help="Path to system prompt file")
    p.add_argument("--working-dir", default=".", help="Working directory for sessions")
    p.add_argument("--claude-command", default="claude", help="Claude CLI binary name")
    p.add_argument("--log-dir", default="./logs", help="Session log directory")
    a = p.parse_args()
    run(Config(**vars(a)))


if __name__ == "__main__":
    main()
