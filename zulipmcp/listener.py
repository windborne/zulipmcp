"""Listen for Zulip @mentions and spawn one agent session per (stream, topic).

With ``--dms``, direct messages to the bot also spawn one session per DM
conversation (DMs from other bots are ignored to avoid reply loops).

Defaults:
- `./.zuliprc` from the current working directory
- `./.mcp.json` from the current working directory (if present)
- bundled `default_system_prompt.md` (resolved relative to this file)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zulip

from .agent_backends import build_agent_cmd, build_agent_env
from . import core as zulip_core

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
    backend: str = "claude"
    agent_command: str | None = None
    mcp_config: Path = Path(".mcp.json")
    system_prompt: Path = DEFAULT_SYSTEM_PROMPT_PATH
    working_dir: Path = Path(".")
    log_dir: Path = Path("./logs")
    codex_permission_mode: str = "parity"
    opencode_model: str = ""
    opencode_agent: str = ""
    dms: bool = False
    backend_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.backend not in {"claude", "codex", "opencode"}:
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        if self.agent_command is None:
            self.agent_command = self.backend
        self.zuliprc = Path(self.zuliprc)
        self.mcp_config = Path(self.mcp_config)
        self.system_prompt = Path(self.system_prompt)
        self.working_dir = Path(self.working_dir)
        self.log_dir = Path(self.log_dir)


# One session per ("stream", stream, topic) or ("dm", peer emails). Value is the Popen object.
_sessions: dict[tuple[str, ...], subprocess.Popen] = {}
_lock = threading.Lock()


def _slug(text: str) -> str:
    """Sanitize user-controlled names for filenames."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._") or "untitled"
    return slug[:100]


def _spawn(cfg: Config, msg: dict[str, Any], dm: list[str] | None = None) -> None:
    """Launch an agent session for this message, if not already running.

    ``dm`` is the other participants' emails for a DM message; stream
    messages use their stream/topic.
    """
    if dm:
        stream = topic = ""
        key: tuple[str, ...] = ("dm", ",".join(dm))
        where = "DM with " + ", ".join(dm)
    else:
        stream, topic = msg["display_recipient"], msg["subject"]
        key = ("stream", stream.lower(), topic.lower())
        where = f"#{stream} > {topic}"

    with _lock:
        p = _sessions.get(key)
        if p and p.poll() is None:
            return  # session alive

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = cfg.log_dir / f"{ts}_{'_'.join(_slug(k) for k in key)}.jsonl"
    fh = open(log_path, "w")
    try:
        env = build_agent_env(cfg, msg)
        cmd = build_agent_cmd(cfg, stream, topic, env, dm=dm)
        proc = subprocess.Popen(
            cmd,
            cwd=cfg.working_dir, env=env,
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"type": "launcher.error", "backend": cfg.backend, "error": str(exc)}
            ),
            file=fh,
        )
        fh.close()
        print(
            f"[listener] Failed to spawn {cfg.backend} session for {where}: "
            f"{exc} (see {log_path})",
            file=sys.stderr,
        )
        return
    with _lock:
        _sessions[key] = proc

    def reap():
        proc.wait()
        fh.close()
        with _lock:
            _sessions.pop(key, None)

    threading.Thread(target=reap, daemon=True, name=f"reap:{'/'.join(key)}").start()
    print(
        f"[listener] Spawned {cfg.backend} session for {where} (pid={proc.pid})",
        file=sys.stderr,
    )


def run(cfg: Config):
    """Block forever, dispatching @mentions to _spawn()."""
    if not cfg.zuliprc.exists():
        raise FileNotFoundError(
            f"Zulip config not found: {cfg.zuliprc} "
            "(set --zuliprc or place .zuliprc in the current working directory)"
        )
    if cfg.backend_flags:
        saved = cfg.backend_flags
        try:
            cfg.backend_flags = []
            base_env = build_agent_env(cfg, {"id": 0, "sender_email": ""})
            base_cmd = build_agent_cmd(cfg, "_", "_", base_env)
            for flag in saved:
                if flag.startswith("--") and flag in base_cmd:
                    print(f"[listener] WARNING: passthrough flag '{flag}' conflicts with a "
                          f"hardcoded flag and may cause unexpected behavior", file=sys.stderr)
        except Exception as exc:
            print(
                f"[listener] WARNING: could not check passthrough flag conflicts: {exc}",
                file=sys.stderr,
            )
        finally:
            cfg.backend_flags = saved
    client = zulip.Client(config_file=str(cfg.zuliprc))
    me = client.get_profile()["user_id"]
    print(
        f"[listener] Listening as user_id={me}, backend={cfg.backend}, log_dir={cfg.log_dir}",
        file=sys.stderr,
    )

    def _is_subscribed(stream_name: str) -> bool:
        """Check if the bot is subscribed to a stream."""
        result = client.get_subscriptions()
        if result.get("result") != "success":
            return False
        return any(
            sub["name"].lower() == stream_name.lower()
            for sub in result.get("subscriptions", [])
        )

    def _is_stream_private(stream_name: str) -> bool:
        """Check if a stream is private (invite_only) via the Zulip API."""
        result = client.get_streams(include_public=True, include_subscribed=True)
        if result.get("result") != "success":
            return True  # can't determine — assume private (safer)
        for s in result.get("streams", []):
            if s["name"].lower() == stream_name.lower():
                return s.get("invite_only", False)
        return True  # stream not in results — must be private (all public streams are always returned)

    def _ensure_subscribed(stream_name: str) -> bool:
        """Subscribe the bot to a public stream. Returns True on success."""
        result = client.add_subscriptions(streams=[{"name": stream_name}])
        return result.get("result") == "success"

    bot_senders: dict[int, bool] = {}

    def _is_bot(user_id: int) -> bool:
        """Whether a user is a bot. Unknown users count as bots (safer)."""
        if user_id not in bot_senders:
            result = client.get_user_by_id(user_id)
            if result.get("result") != "success":
                return True
            bot_senders[user_id] = result["user"].get("is_bot", False)
        return bot_senders[user_id]

    def handle_dm(msg: dict[str, Any]) -> None:
        if msg.get("sender_id") == me or _is_bot(msg["sender_id"]):
            return
        peers = sorted(zulip_core.dm_peers(msg, me) or ())
        if not peers:
            return
        try:
            client.add_reaction({"message_id": msg["id"], "emoji_name": "eyes"})
        except Exception:
            pass
        _spawn(cfg, msg, dm=peers)

    def handle(event):
        msg = event["message"]
        if cfg.dms and msg.get("type") == "private":
            handle_dm(msg)
            return
        if (msg.get("type") == "stream"
                and "mentioned" in event.get("flags", [])
                and msg.get("sender_id") != me):
            stream = msg["display_recipient"]
            topic = msg["subject"]

            if not _is_subscribed(stream):
                if _is_stream_private(stream):
                    # Private stream the bot wasn't invited to — send error
                    if zulip_core.is_stream_write_allowed(stream):
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

                # Public stream — auto-subscribe so the session can read/listen
                if _ensure_subscribed(stream):
                    print(f"[listener] Auto-subscribed to public stream #{stream}", file=sys.stderr)
                else:
                    print(f"[listener] Failed to auto-subscribe to #{stream}", file=sys.stderr)

            try:
                client.add_reaction({"message_id": msg["id"], "emoji_name": "eyes"})
            except Exception:
                pass
            _spawn(cfg, msg)

    client.call_on_each_event(handle, event_types=["message"])


def main():
    p = argparse.ArgumentParser(
        description="Zulip agent listener",
        epilog="See zulipmcp README for full setup instructions.",
    )
    p.add_argument("--zuliprc", default=".zuliprc",
                   help="Path to .zuliprc (default: ./.zuliprc)")
    p.add_argument("--backend", choices=["claude", "codex", "opencode"], default="claude",
                   help="Agent backend to spawn (default: claude)")
    p.add_argument("--agent-command", default=None,
                   help="Agent CLI binary name/path (default: backend name)")
    p.add_argument("--mcp-config", default=".mcp.json",
                   help="Path to .mcp.json for agent sessions (default: ./.mcp.json if present)")
    p.add_argument(
        "--system-prompt",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help=(
            "Path to system prompt file "
            f"(default: {DEFAULT_SYSTEM_PROMPT_PATH.name} bundled with zulipmcp)"
        ),
    )
    p.add_argument("--working-dir", default=".", help="Working directory for sessions")
    p.add_argument("--log-dir", default="./logs", help="Session log directory")
    p.add_argument(
        "--codex-permission-mode",
        choices=["parity", "workspace-write", "read-only", "none"],
        default="parity",
        help=(
            "Codex permission preset: parity uses --yolo for full bypass like the Claude default; "
            "workspace-write/read-only use noninteractive sandboxed modes; none adds no flags "
            "(default: parity)"
        ),
    )
    p.add_argument(
        "--opencode-model", default="",
        help="OpenCode model in provider/model format (e.g. 'anthropic/claude-sonnet-4-5')",
    )
    p.add_argument(
        "--opencode-agent", default="",
        help="OpenCode agent name (passed as --agent)",
    )
    p.add_argument(
        "--dms", action="store_true",
        help="Also spawn a session for direct messages to the bot (off by default)",
    )
    p.add_argument(
        "backend_flags", nargs=argparse.REMAINDER,
        help="Additional flags forwarded to the selected backend (place after --)",
    )
    a = p.parse_args()
    flags = a.backend_flags or []
    if flags and flags[0] == "--":
        flags = flags[1:]
    a.backend_flags = flags
    run(Config(**vars(a)))


if __name__ == "__main__":
    main()
