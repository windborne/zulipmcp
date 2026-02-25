"""Listen for Zulip @mentions and spawn one Claude session per (stream, topic).

Defaults:
- `./.zuliprc` from the current working directory
- `./.mcp.json` from the current working directory (if present)
- bundled `default_system_prompt.md` (resolved relative to this file)
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import zulip

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PACKAGE_DIR / "default_system_prompt.md"

PRIVATE_STREAM_NOT_INVITED = (
    "I don't have access to this private stream. "
    "To use me here, a stream admin needs to invite me first. "
    "You can do this from the stream settings \u2192 Subscribers \u2192 add the bot."
)


@dataclass
class Config:
    """CLI args map 1:1 to fields."""
    zuliprc: Path
    mcp_config: Path = Path(".mcp.json")
    system_prompt: Path = DEFAULT_SYSTEM_PROMPT_PATH
    working_dir: Path = Path(".")
    claude_command: str = "claude"
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        self.zuliprc = Path(self.zuliprc)
        self.mcp_config = Path(self.mcp_config)
        self.system_prompt = Path(self.system_prompt)
        self.working_dir = Path(self.working_dir)
        self.log_dir = Path(self.log_dir)


# One session per (stream, topic). Value is the Popen object.
_sessions: dict[tuple[str, str], subprocess.Popen] = {}
_lock = threading.Lock()


def _slug(text: str) -> str:
    """Sanitize user-controlled names for filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._") or "untitled"


def _build_cmd(cfg: Config, stream: str, topic: str) -> list[str]:
    """Assemble the claude CLI invocation."""
    cmd = [cfg.claude_command, "--dangerously-skip-permissions",
           "--output-format", "stream-json", "--verbose"]
    if cfg.mcp_config.exists():
        cmd += ["--mcp-config", str(cfg.mcp_config.resolve())]
    if cfg.system_prompt.exists():
        cmd += ["--append-system-prompt", cfg.system_prompt.read_text()]
    cmd += ["-p", f"Call set_context('{stream}', '{topic}') to begin, "
                   f"then greet the user and listen for follow-ups."]
    return cmd


def _build_env(cfg: Config, msg: dict) -> dict:
    """Set env vars that zulipmcp reads on the other side."""
    env = os.environ.copy()
    env["ZULIP_RC_PATH"] = str(cfg.zuliprc.resolve())
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

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = cfg.log_dir / f"{ts}_{_slug(key[0])}_{_slug(key[1])}.jsonl"
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
    if not cfg.zuliprc.exists():
        raise FileNotFoundError(
            f"Zulip config not found: {cfg.zuliprc} (expected .zuliprc in the current working directory)"
        )
    client = zulip.Client(config_file=str(cfg.zuliprc))
    me = client.get_profile()["user_id"]
    print(f"[listener] Listening as user_id={me}, log_dir={cfg.log_dir}", file=sys.stderr)

    def _is_subscribed(stream_name: str) -> bool:
        """Check if the bot is subscribed to a stream."""
        result = client.get_subscriptions()
        if result.get("result") != "success":
            return False
        return any(
            sub["name"].lower() == stream_name.lower()
            for sub in result.get("subscriptions", [])
        )

    def handle(event):
        msg = event["message"]
        if (msg.get("type") == "stream"
                and "mentioned" in event.get("flags", [])
                and msg.get("sender_id") != me):
            stream = msg["display_recipient"]
            topic = msg["subject"]

            # If the bot isn't subscribed, it can't read messages in this
            # stream (private stream it wasn't invited to).  Send a canned
            # error and skip spawning a session — no LLM needed.
            if not _is_subscribed(stream):
                try:
                    client.send_message({
                        "type": "stream",
                        "to": stream,
                        "topic": topic,
                        "content": PRIVATE_STREAM_NOT_INVITED,
                    })
                except Exception as e:
                    print(f"[listener] Failed to send private-stream error to #{stream} > {topic}: {e}",
                          file=sys.stderr)
                return

            try:
                client.add_reaction({"message_id": msg["id"], "emoji_name": "eyes"})
            except Exception:
                pass
            _spawn(cfg, msg)

    client.call_on_each_event(handle, event_types=["message"])


def main():
    p = argparse.ArgumentParser(
        description="Zulip → Claude Code listener",
        epilog="See zulipmcp README for full setup instructions.",
    )
    p.add_argument("--zuliprc", default=".zuliprc",
                   help="Path to .zuliprc (default: ./.zuliprc)")
    p.add_argument("--mcp-config", default=".mcp.json",
                   help="Path to .mcp.json for Claude (default: ./.mcp.json if present)")
    p.add_argument(
        "--system-prompt",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help=(
            "Path to system prompt file "
            f"(default: {DEFAULT_SYSTEM_PROMPT_PATH.name} bundled with zulipmcp)"
        ),
    )
    p.add_argument("--working-dir", default=".", help="Working directory for sessions")
    p.add_argument("--claude-command", default="claude", help="Claude CLI binary name")
    p.add_argument("--log-dir", default="./logs", help="Session log directory")
    a = p.parse_args()
    run(Config(**vars(a)))


if __name__ == "__main__":
    main()
