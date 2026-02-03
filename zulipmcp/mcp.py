#!/usr/bin/env python3
"""Zulip MCP Server — thin wrapper over zulipmcp.core for Claude Code / MCP clients.

All Zulip logic lives in zulipmcp.core; this file only handles MCP tool
registration and session state.

Usage:
    python -m zulipmcp.mcp                  # stdio (Claude Code)
    python -m zulipmcp.mcp --transport sse  # SSE
"""

import time
import asyncio
from typing import Optional
from dataclasses import dataclass

from fastmcp import FastMCP
from fastmcp.server.context import Context

from . import core as zulip_core

mcp = FastMCP("Zulip Messaging")


# ============================================================================
# Hook system — allows callers to customize MCP behavior without forking
# ============================================================================

_hooks: dict = {
    "message_prefix": None,   # () -> str : prepended to reply() and send_message()
    "on_session_end": None,   # () -> None : called when end_session() runs
    "on_set_context": None,   # (stream, topic) -> str : extra text appended to set_context response
}


def configure(**kwargs):
    """Configure optional hooks for the MCP server.

    Supported hooks:
        message_prefix: callable() -> str
            Returns a prefix string prepended to all outgoing messages
            (reply and send_message). Return "" to skip.
        on_session_end: callable() -> None
            Called when end_session() is invoked. Use for cleanup
            (e.g. writing exit markers).
        on_set_context: callable(stream: str, topic: str) -> str
            Returns extra text to append to the set_context response
            (e.g. custom instructions, prompts).
    """
    for key, value in kwargs.items():
        if key not in _hooks:
            raise ValueError(f"Unknown hook: {key!r}. Valid hooks: {list(_hooks.keys())}")
        _hooks[key] = value


def _get_prefix() -> str:
    """Get message prefix from hook, or empty string."""
    fn = _hooks.get("message_prefix")
    if fn:
        try:
            return fn()
        except Exception:
            return ""
    return ""


@dataclass
class SessionState:
    """Tracks the current conversation session."""
    stream: Optional[str] = None
    topic: Optional[str] = None
    last_seen_message_id: Optional[int] = None
    my_user_id: Optional[int] = None
    active: bool = False

    def reset(self):
        self.stream = None
        self.topic = None
        self.last_seen_message_id = None
        self.active = False


_session = SessionState()


# ============================================================================
# Session tools — for interactive chat participation
# ============================================================================

@mcp.tool()
def set_context(stream: str, topic: str) -> str:
    """Set the session context for a conversation. Call once at the start.

    Args:
        stream: Stream/channel name.
        topic: Topic name within the stream.
    """
    profile = zulip_core.get_profile()
    if profile.get("result") == "success":
        _session.my_user_id = profile.get("user_id")

    _session.stream = stream
    _session.topic = topic
    _session.active = True

    messages = zulip_core.get_topic_messages(stream, topic, num_messages=20)
    if messages:
        _session.last_seen_message_id = messages[-1]["id"]

    header = f"Session context set to #{stream} > {topic}"

    # Allow hook to inject extra context (e.g. custom instructions)
    on_set_ctx = _hooks.get("on_set_context")
    if on_set_ctx:
        try:
            extra = on_set_ctx(stream, topic)
            if extra:
                header += "\n\n" + extra
        except Exception:
            pass

    header += "\n\nRecent messages:"
    return header + "\n\n" + zulip_core.format_messages(messages, include_topic=False)


@mcp.tool()
def reply(content: str) -> str:
    """Reply in the current session context.

    Args:
        content: Message content (supports Zulip markdown).
    """
    if not _session.active or not _session.stream or not _session.topic:
        return "Error: No session context set. Call set_context first."

    # Check for missed messages before sending
    missed = []
    if _session.last_seen_message_id:
        missed = zulip_core.fetch_new_messages(
            _session.stream, _session.topic,
            _session.last_seen_message_id, _session.my_user_id,
        )

    prefix = _get_prefix()
    result = zulip_core.send_message(_session.stream, _session.topic, prefix + content)
    if result["result"] != "success":
        return f"Error sending message: {result.get('msg', 'Unknown error')}"

    sent_id = result.get("id")
    _session.last_seen_message_id = sent_id
    response = f"Message sent (id: {sent_id})"

    if missed:
        _session.last_seen_message_id = missed[-1]["id"]
        response += "\n\nMissed messages:\n\n" + zulip_core.format_messages(missed)

    response += "\n\n[Hint: Use `listen` tool if you need to wait for a reply.]"
    return response


@mcp.tool()
async def listen(timeout_hours: float, ctx: Context) -> str:
    """Wait for new messages in the current context (blocking).

    Args:
        timeout_hours: Max wait time in hours. Default to 1.
    """
    if not _session.active or not _session.stream or not _session.topic:
        return "Error: No session context set. Call set_context first."

    timeout_seconds = timeout_hours * 3600

    # Save the message ID before the loop — the session field gets updated
    # when new messages arrive, so we need the original to remove the emoji.
    listen_msg_id = _session.last_seen_message_id

    # Add listening indicator
    if listen_msg_id:
        try:
            zulip_core.add_reaction(listen_msg_id, "robot_ear")
        except Exception:
            pass

    try:
        start = time.time()
        end = start + timeout_seconds
        last_heartbeat = start

        while time.time() < end:
            messages = zulip_core.fetch_new_messages(
                _session.stream, _session.topic,
                _session.last_seen_message_id, _session.my_user_id,
            )
            if messages:
                _session.last_seen_message_id = messages[-1]["id"]
                return "New messages:\n\n" + zulip_core.format_messages(messages)

            now = time.time()
            if now - last_heartbeat >= 10:
                elapsed = int(now - start)
                # Use ctx.info() as keep-alive — report_progress is a no-op
                # when the client doesn't send a progressToken, which causes
                # the MCP connection to timeout during long polls.
                await ctx.info(f"Listening… {elapsed}s elapsed")
                await ctx.report_progress(progress=elapsed, total=timeout_seconds)
                last_heartbeat = now

            await asyncio.sleep(2)

        return (
            f"Timeout: No new messages after {timeout_hours} hours.\n\n"
            "[Hint: Send a check-in message, then wait again with exponential backoff.]"
        )
    finally:
        if listen_msg_id:
            try:
                zulip_core.remove_reaction(listen_msg_id, "robot_ear")
            except Exception:
                pass


@mcp.tool()
def end_session() -> str:
    """End the current session gracefully."""
    if not _session.active:
        return "No active session to end."
    info = f"Session ended. Was chatting in #{_session.stream} > {_session.topic}"

    on_end = _hooks.get("on_session_end")
    if on_end:
        try:
            on_end()
        except Exception:
            pass

    _session.reset()
    return info


# ============================================================================
# Read tools — browsing and searching
# ============================================================================

@mcp.tool()
def list_streams(include_private: bool = False) -> str:
    """List all available Zulip streams/channels.

    Args:
        include_private: Include private streams the bot can access.
    """
    streams = zulip_core.list_streams(include_private=include_private)
    if not streams:
        return "No streams found."
    lines = []
    for s in streams:
        line = f"- #{s['name']}"
        if s.get("description"):
            line += f": {s['description']}"
        lines.append(line)
    return f"Found {len(streams)} streams:\n\n" + "\n".join(lines)


@mcp.tool()
def get_stream_topics(stream: str, limit: int = 20) -> str:
    """Get recent topics in a stream.

    Args:
        stream: Stream/channel name.
        limit: Max topics to return (default 20).
    """
    try:
        topics = zulip_core.get_stream_topics(stream, limit=limit)
    except ValueError as e:
        return f"Error: {e}"
    if not topics:
        return f"No topics found in #{stream}."
    lines = [f"Recent topics in #{stream}:"]
    for t in topics:
        lines.append(f"- {t.get('name', 'Unknown')}")
    return "\n".join(lines)


@mcp.tool()
def get_stream_members(stream: str) -> str:
    """Get the members of a stream/channel.

    Args:
        stream: Stream/channel name.
    """
    try:
        members = zulip_core.get_stream_members(stream)
    except ValueError as e:
        return f"Error: {e}"
    if not members:
        return f"No members found in #{stream}."
    lines = [f"#{stream} has {len(members)} member(s):"]
    for u in members:
        lines.append(f"- {u['full_name']} ({u['email']})")
    return "\n".join(lines)


@mcp.tool()
def get_messages(stream: str, topic: str, num_messages: int = 20,
                 before_message_id: Optional[int] = None) -> str:
    """Get messages from a specific stream and topic.

    Args:
        stream: Stream/channel name.
        topic: Topic name.
        num_messages: Number of messages (default 20, max 100).
        before_message_id: Get messages before this ID (for pagination).
    """
    messages = zulip_core.get_topic_messages(
        stream, topic, num_messages=num_messages,
        before_message_id=before_message_id,
    )
    header = f"Messages from #{stream} > {topic}:"
    return header + "\n\n" + zulip_core.format_messages(messages)


@mcp.tool()
def get_message_by_id(message_id: int) -> str:
    """Get a specific message by its ID.

    Args:
        message_id: The message ID.
    """
    msg = zulip_core.get_message_by_id(message_id)
    if not msg:
        return f"Message {message_id} not found."
    return zulip_core.format_messages([msg], include_topic=True)


@mcp.tool()
def get_message_link(message_id: int) -> str:
    """Get a permalink for a Zulip message.

    Args:
        message_id: The message ID.
    """
    return zulip_core.get_message_link(message_id)


# ============================================================================
# Write tools — sending messages and reactions
# ============================================================================

@mcp.tool()
def send_message(stream: str, topic: str, content: str) -> str:
    """Send a message to a specific stream and topic (fire-and-forget).

    Args:
        stream: Stream/channel name.
        topic: Topic name.
        content: Message content (supports Zulip markdown).
    """
    prefix = _get_prefix()
    result = zulip_core.send_message(stream, topic, prefix + content)
    if result["result"] != "success":
        return f"Error sending message: {result.get('msg', 'Unknown error')}"
    return f"Message sent to #{stream} > {topic} (id: {result.get('id')})"


@mcp.tool()
def add_reaction(message_id: int, emoji_name: str) -> str:
    """Add an emoji reaction to a message.

    Args:
        message_id: The message ID.
        emoji_name: Emoji name without colons (e.g. "thumbs_up", "check").
    """
    result = zulip_core.add_reaction(message_id, emoji_name)
    if result["result"] != "success":
        return f"Error adding reaction: {result.get('msg', 'Unknown error')}"
    return f"Added :{emoji_name}: to message {message_id}"


# ============================================================================
# Info tools — users, subscriptions
# ============================================================================

@mcp.tool()
def get_user_info(email: str) -> str:
    """Get information about a Zulip user.

    Args:
        email: The user's email address.
    """
    user = zulip_core.get_user_info(email)
    if not user:
        return f"User not found: {email}"
    lines = [
        f"User: {user.get('full_name', 'Unknown')}",
        f"Email: {user.get('email', email)}",
        f"User ID: {user.get('user_id', 'Unknown')}",
        f"Role: {user.get('role', 'Unknown')}",
        f"Active: {user.get('is_active', 'Unknown')}",
    ]
    if user.get("timezone"):
        lines.append(f"Timezone: {user['timezone']}")
    return "\n".join(lines)


@mcp.tool()
def resolve_name(query: str) -> str:
    """Look up a user's display name by substring before mentioning them.

    Call this BEFORE using @**Name** in a message if you're not 100% sure of the
    exact display name. Zulip mentions require an exact match.

    Args:
        query: Substring to search for (case-insensitive). e.g. "john", "smith".
    """
    matches = zulip_core.resolve_name(query)
    if not matches:
        return f"No active users matching \"{query}\"."
    lines = [f"Found {len(matches)} user(s) matching \"{query}\":"]
    for u in matches:
        lines.append(f"- {u['full_name']} ({u['email']})")
    if len(matches) == 1:
        lines.append(f"\nTo mention them: @**{matches[0]['full_name']}**")
    return "\n".join(lines)


@mcp.tool()
def get_subscribed_streams() -> str:
    """Get streams the bot is subscribed to."""
    subs = zulip_core.get_subscribed_streams()
    if not subs:
        return "Not subscribed to any streams."
    lines = [f"Subscribed to {len(subs)} streams:"]
    for s in subs:
        lines.append(f"- #{s['name']}")
    return "\n".join(lines)


# ============================================================================
# File tools — downloading uploads
# ============================================================================

@mcp.tool()
def fetch_image(path: str) -> str:
    """Fetch an image from Zulip and save it to a temp file for viewing.

    Args:
        path: Image path from message content (e.g. "/user_uploads/2/54/abc/image.jpg").
    """
    try:
        temp_path = zulip_core.save_image(path)
    except ValueError as e:
        return f"Error fetching image: {e}"
    return f"Image saved to: {temp_path}\n\nUse the Read tool to view the image at this path."


@mcp.tool()
def fetch_file(path: str, save_dir: Optional[str] = None) -> str:
    """Fetch any file from Zulip and save it locally.

    Args:
        path: File path from message content (e.g. "/user_uploads/...").
        save_dir: Directory to save to. Uses temp dir if not provided.
    """
    try:
        saved_path, size_bytes, content_type = zulip_core.save_file(path, save_dir=save_dir)
    except ValueError as e:
        return f"Error fetching file: {e}"
    return f"File saved to: {saved_path}\nSize: {size_bytes / 1024:.1f} KB\nContent-Type: {content_type}"


# ============================================================================
# Server entry point
# ============================================================================

def run_server(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8235):
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zulip MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8235)
    args = parser.parse_args()
    run_server(transport=args.transport, port=args.port)
