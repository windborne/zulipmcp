import os
import time
import tempfile
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import diskcache
import zulip

TIMEZONE = ZoneInfo("America/Los_Angeles")

_client: Optional[zulip.Client] = None
_cache = diskcache.Cache(os.environ.get("ZULIPMCP_CACHE_DIR",
    Path(tempfile.gettempdir()) / "zulipmcp_cache"))
_ignored_streams: set[str] = set()
_ALL_PRIVATE_STREAMS = "__ALL__"


def _allowed_private_streams() -> set[str]:
    """Return allowed private streams from env.

    Security model:
    - Unset/empty env => no private stream access (default-deny)
    - "__ALL__" => explicit full private stream access
    - list/string => explicit allowlist (normalized to lowercase)
    """
    raw = os.environ.get("BOT_ALLOWED_PRIVATE_STREAMS", "").strip()
    if not raw:
        return set()
    if raw == _ALL_PRIVATE_STREAMS:
        return {_ALL_PRIVATE_STREAMS}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {str(s).lower() for s in parsed if str(s).strip()}
        if isinstance(parsed, str):
            if parsed == _ALL_PRIVATE_STREAMS:
                return {_ALL_PRIVATE_STREAMS}
            return {parsed.lower()}
    except json.JSONDecodeError:
        # Comma-separated fallback
        return {s.strip().lower() for s in raw.split(",") if s.strip()}
    return set()


def is_private_stream_allowed(stream_name: str) -> bool:
    """Whether the current process is allowed to access this private stream."""
    if not is_stream_private(stream_name):
        return True
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS in allowed:
        return True
    return stream_name.lower() in allowed


def set_ignored_streams(streams: set[str]) -> None:
    """Set streams to exclude from all message fetches."""
    global _ignored_streams
    _ignored_streams = {s.lower() for s in streams}


def get_ignored_streams() -> set[str]:
    """Return the current ignored streams set."""
    return _ignored_streams


def is_stream_private(stream_name: str) -> bool:
    """Check if a stream is private (invite_only). Cached for 1 hour."""
    cache_key = ("is_stream_private", stream_name.lower())
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch all streams and cache every one — avoids repeated API calls
    result = get_client().get_streams(include_public=True, include_subscribed=True)
    if result["result"] != "success":
        return False

    for s in result.get("streams", []):
        is_private = s.get("invite_only", False)
        _cache.set(("is_stream_private", s["name"].lower()), is_private, expire=3600)

    # Re-check cache after populating
    cached = _cache.get(cache_key)
    return cached if cached is not None else False


def get_client() -> zulip.Client:
    """Get or create the Zulip client singleton.

    Config resolution order:
        1. ZULIP_RC_PATH environment variable (absolute path to zuliprc)
        2. .zuliprc in the current working directory
    """
    global _client
    if _client is None:
        import os
        import sys
        rc_env = os.environ.get("ZULIP_RC_PATH")
        if rc_env:
            config_path = Path(rc_env)
        else:
            config_path = Path.cwd() / ".zuliprc"
        if not config_path.exists():
            raise FileNotFoundError(
                f"zuliprc not found at: {config_path}\n"
                f"Set ZULIP_RC_PATH env var or place .zuliprc in the working directory."
            )
        _client = zulip.Client(config_file=str(config_path))
        # Log identity so it's visible in MCP startup output
        email = _client.email or "unknown"
        print(f"[zulipmcp] Zulip client initialized as: {email} (from {config_path})", file=sys.stderr)
    return _client


def get_user_email(full_name: str) -> Optional[str]:
    """Look up a user's email by their full name."""
    client = get_client()
    result = client.get_users()
    if result["result"] != "success":
        return None
    for user in result["members"]:
        if user["full_name"] == full_name:
            return user["email"]
    return None


def resolve_name(query: str) -> list[dict]:
    """Find active users whose display name contains the query (case-insensitive).

    Returns list of dicts with 'full_name' and 'email' for each match.
    """
    client = get_client()
    result = client.get_users()
    if result["result"] != "success":
        return []
    q = query.lower()
    matches = []
    for user in result["members"]:
        if user.get("is_bot", False) or not user.get("is_active", True):
            continue
        if q in user["full_name"].lower():
            matches.append({"full_name": user["full_name"], "email": user["email"]})
    matches.sort(key=lambda u: u["full_name"])
    return matches


# ============================================================================
# Bot visibility filtering — /nobots and /nb support
# ============================================================================

_NOBOTS_KEYWORDS = ("/nobots", "/nb")


def _should_hide_from_bot(msg: dict) -> bool:
    """Check if a message should be hidden from bots.

    Returns True if:
    - The topic contains '/nobots' or '/nb' (case-insensitive)
    - The message content starts with '/nobots' or '/nb' (case-insensitive, after stripping)
    """
    topic = msg.get("subject", "").lower()
    if any(kw in topic for kw in _NOBOTS_KEYWORDS):
        return True
    content = msg.get("content", "").strip().lower()
    if any(content.startswith(kw) for kw in _NOBOTS_KEYWORDS):
        return True
    return False


def filter_for_bot(messages: list[dict]) -> list[dict]:
    """Filter out messages that should be hidden from bots.

    Use this before displaying messages to the bot or checking if new messages exist.
    """
    return [m for m in messages if not _should_hide_from_bot(m)]


def _format_timestamp(timestamp: int, prev_timestamp: Optional[int] = None) -> str:
    """Smart timestamp formatting.

    - Always shows date on the first message
    - Shows date only when it changes from the previous message
    - Shows time only when there's a 5+ minute gap from the previous message
    - Uses PT timezone
    """
    dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
    show_date = True
    show_time = True

    if prev_timestamp is not None:
        prev_dt = datetime.fromtimestamp(prev_timestamp, tz=TIMEZONE)
        show_date = dt.date() != prev_dt.date()
        gap_minutes = (timestamp - prev_timestamp) / 60
        show_time = gap_minutes >= 5 or show_date

    if show_date and show_time:
        return dt.strftime("%Y-%m-%d %H:%M:%S PT")
    elif show_time:
        return dt.strftime("%H:%M:%S PT")
    else:
        return ""


def _time_attr(msg: dict, prev_timestamp: Optional[int]) -> str:
    """Build the time attribute string for a <msg> tag."""
    timestamp = msg.get("timestamp", 0)
    if "time_range_end" in msg:
        start = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        end = datetime.fromtimestamp(msg["time_range_end"], tz=TIMEZONE)
        if start.date() != end.date():
            return f'time="{start.strftime("%Y-%m-%d %H:%M:%S")}-{end.strftime("%Y-%m-%d %H:%M:%S")} PT"'
        return f'time="{start.strftime("%Y-%m-%d %H:%M:%S")}-{end.strftime("%H:%M:%S")} PT"'
    ts_str = _format_timestamp(timestamp, prev_timestamp)
    if ts_str:
        return f'time="{ts_str}"'
    return ""


def get_messages(hours_back: int = 24, channels: Optional[list[str]] = None,
                 sender: Optional[str] = None,
                 as_of: Optional[datetime] = None) -> list[dict]:
    """General-purpose paginated fetch returning raw message dicts sorted by ID.

    channels=None → all public streams
    channels=["eng", "ops"] → specific streams
    sender="John Dean" → auto-resolves name to email, filters by sender
    as_of=datetime(2025,6,1) → treat this as "now" (fetch hours_back before it)
    """
    cache_key = ("get_messages", hours_back, tuple(channels) if channels else None,
                 sender, as_of.isoformat() if as_of else None)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_client()
    anchor_time = int(as_of.timestamp()) if as_of else int(time.time())
    cutoff = anchor_time - (hours_back * 3600)

    narrow = []
    if channels is None:
        narrow.append({"operator": "channels", "operand": "public"})
    # If specific channels given, we fetch per-channel and merge
    if sender:
        email = get_user_email(sender)
        if not email:
            return []
        narrow.append({"operator": "sender", "operand": email})

    if channels is not None:
        # Fetch from each channel separately and merge
        all_messages = []
        for channel in channels:
            channel_narrow = narrow + [{"operator": "channel", "operand": channel}]
            msgs = _paginated_fetch(client, channel_narrow, cutoff, anchor_time)
            all_messages.extend(msgs)
        # Deduplicate by id and sort
        seen = set()
        unique = []
        for m in all_messages:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique.append(m)
        unique.sort(key=lambda m: m["id"])
        _cache.set(cache_key, unique, expire=600)
        return unique
    else:
        result = _paginated_fetch(client, narrow, cutoff, anchor_time)
        _cache.set(cache_key, result, expire=600)
        return result


def _paginated_fetch(client: zulip.Client, narrow: list[dict],
                     cutoff: int, upper_bound: Optional[int] = None) -> list[dict]:
    """Paginate backward from newest in batches of 500 until time cutoff."""
    cache_key = ("_paginated_fetch", tuple(tuple(d.items()) for d in narrow), cutoff, upper_bound)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    all_messages = []
    anchor = "newest"
    while True:
        result = client.get_messages({
            "anchor": anchor,
            "num_before": 500,
            "num_after": 0,
            "narrow": narrow,
            "apply_markdown": False,
        })
        if result["result"] != "success":
            break
        batch = result["messages"]
        if not batch:
            break

        found_old = False
        for msg in batch:
            if upper_bound and msg["timestamp"] > upper_bound:
                continue
            if msg["timestamp"] >= cutoff:
                all_messages.append(msg)
            else:
                found_old = True

        if found_old:
            break

        # Move anchor to oldest message in this batch
        anchor = batch[0]["id"]
        # If we got fewer than requested, we've hit the beginning
        if len(batch) < 500:
            break

    all_messages.sort(key=lambda m: m["id"])
    if _ignored_streams:
        all_messages = [
            m for m in all_messages
            if m.get("display_recipient", "").lower() not in _ignored_streams
        ]
    _cache.set(cache_key, all_messages, expire=600)
    return all_messages


def format_messages(messages: list[dict], include_topic: bool = False,
                    combine: bool = True) -> str:
    """Format messages as XML <msg> tags.

    combine=True merges consecutive messages from same user within 2 min.
    Smart timestamps: date shown only when it changes, time shown on 5+ min gaps.
    include_topic=True adds stream and topic attributes to each <msg> tag.

    Note: Messages in /nobots or /nb topics or starting with /nobots or /nb are automatically
    filtered out and will not be shown to the bot.
    """
    if not messages:
        return ""

    # Filter out messages that should be hidden from bots
    messages = filter_for_bot(messages)
    if not messages:
        return ""

    if combine:
        messages = _combine_messages(messages)

    lines = []
    prev_timestamp = None
    for msg in messages:
        sender = msg.get("sender_full_name", "Unknown")
        msg_id = msg.get("id", "")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", 0)

        # Build the time attribute
        time_attr = _time_attr(msg, prev_timestamp)

        attrs = [f'user="{sender}"']
        if time_attr:
            attrs.append(time_attr)
        if include_topic and msg.get("type") == "stream":
            stream_name = msg.get("display_recipient", "")
            visibility = "private" if is_stream_private(stream_name) else "public"
            attrs.append(f'stream="{stream_name}"')
            attrs.append(f'visibility="{visibility}"')
            attrs.append(f'topic="{msg.get("subject", "")}"')
        attrs.append(f'id="{msg_id}"')

        # Add reaction summary as attribute if present
        reaction_str = summarize_reactions(msg.get("reactions", []))
        if reaction_str:
            attrs.append(f'reactions="{reaction_str}"')

        tag_open = "<msg " + " ".join(attrs) + ">"
        lines.append(f"{tag_open}\n{content}\n</msg>")

        prev_timestamp = msg.get("time_range_end", timestamp)

    return "\n".join(lines)


def _combine_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages from same user within 2 minutes."""
    if not messages:
        return []

    combined = []
    current = dict(messages[0])
    current["reactions"] = list(current.get("reactions", []))

    for msg in messages[1:]:
        same_user = msg.get("sender_full_name") == current.get("sender_full_name")
        time_gap = msg.get("timestamp", 0) - current.get("time_range_end", current.get("timestamp", 0))

        if same_user and time_gap <= 120:
            # Merge: append content, update end time, id, and reactions
            current["content"] = current["content"] + "\n\n" + msg.get("content", "")
            current["time_range_end"] = msg.get("timestamp", 0)
            current["id"] = msg.get("id", current["id"])
            current["reactions"].extend(msg.get("reactions", []))
        else:
            combined.append(current)
            current = dict(msg)
            current["reactions"] = list(current.get("reactions", []))

    combined.append(current)
    return combined


def _context_window(messages: list[dict], sender_email: str) -> list[dict]:
    """Keep messages around where sender participated.

    Before each user message: 3 messages OR 10 min (whichever captures more).
    After each user message: 3 messages OR (2 hours AND <10 messages).
    """
    if not messages:
        return []

    include = set()
    sender_indices = [i for i, m in enumerate(messages)
                      if m.get("sender_email") == sender_email]

    for idx in sender_indices:
        include.add(idx)
        ts = messages[idx]["timestamp"]

        # Before: 3 messages or 10 min
        for i in range(idx - 1, -1, -1):
            if (idx - i) <= 3 or (ts - messages[i]["timestamp"]) <= 600:
                include.add(i)
            else:
                break

        # After: 3 messages or (2hr and <10 messages)
        for i in range(idx + 1, len(messages)):
            dist = i - idx
            if dist <= 3 or (messages[i]["timestamp"] - ts <= 7200 and dist < 10):
                include.add(i)
            else:
                break

    return [messages[i] for i in sorted(include)]


def format_topics(messages: list[dict], combine: bool = True) -> str:
    """Group messages by stream>topic, format each group.

    Topics sorted by last message ID. Groups separated by ---.
    """
    if not messages:
        return ""

    # Group by (stream, topic)
    groups: dict[tuple[str, str], list[dict]] = {}
    for msg in messages:
        stream = msg.get("display_recipient", "unknown")
        topic = msg.get("subject", "unknown")
        key = (stream, topic)
        if key not in groups:
            groups[key] = []
        groups[key].append(msg)

    # Sort groups by last message ID
    sorted_groups = sorted(groups.items(), key=lambda item: item[1][-1]["id"])

    sections = []
    for (stream, topic), group_msgs in sorted_groups:
        visibility = "[private]" if is_stream_private(stream) else "[public]"
        header = f"Stream: {visibility} {stream}, Topic: {topic}"
        body = format_messages(group_msgs, include_topic=False, combine=combine)
        sections.append(f"{header}\n{body}")

    return "\n---\n".join(sections)


def get_messages_formatted(hours_back: int = 24, channels: Optional[list[str]] = None,
                           sender: Optional[str] = None,
                           group_by_topic: bool = True,
                           as_of: Optional[datetime] = None) -> str:
    """Convenience: fetch + format in one call.

    When sender is set with group_by_topic=True, fetches full conversation
    context around the sender's messages in each topic they participated in.
    as_of=datetime(2025,6,1) → treat this as "now" (fetch hours_back before it)
    """
    if sender and group_by_topic:
        return _get_sender_conversations(hours_back, channels, sender, as_of)

    messages = get_messages(hours_back=hours_back, channels=channels, sender=sender, as_of=as_of)
    if not messages:
        return "No messages found."
    if group_by_topic:
        return format_topics(messages)
    else:
        return format_messages(messages, include_topic=True)


def _get_sender_conversations(hours_back: int, channels: Optional[list[str]],
                              sender: str, as_of: Optional[datetime] = None) -> str:
    """Two-pass fetch: find sender's topics, then get full context."""
    client = get_client()
    email = get_user_email(sender)
    if not email:
        return "No messages found."

    # Pass 1: get sender's messages to discover topics
    sender_messages = get_messages(hours_back=hours_back, channels=channels, sender=sender, as_of=as_of)
    if not sender_messages:
        return "No messages found."

    # Extract unique (stream, topic) pairs
    topics = set()
    for msg in sender_messages:
        if msg.get("type") == "stream":
            topics.add((msg["display_recipient"], msg["subject"]))

    # Pass 2: fetch all messages from each topic, apply context window
    anchor_time = int(as_of.timestamp()) if as_of else int(time.time())
    cutoff = anchor_time - (hours_back * 3600)
    all_context_messages = []
    for stream, topic in topics:
        narrow = [
            {"operator": "channel", "operand": stream},
            {"operator": "topic", "operand": topic},
        ]
        topic_messages = _paginated_fetch(client, narrow, cutoff, anchor_time)
        windowed = _context_window(topic_messages, email)
        all_context_messages.extend(windowed)

    if not all_context_messages:
        return "No messages found."

    return format_topics(all_context_messages)


# ============================================================================
# Direct API wrappers — return raw Python objects for composability
# ============================================================================

def get_profile() -> dict:
    """Get the bot's own profile. Returns the profile dict from the API."""
    return get_client().get_profile()


def list_streams(include_private: bool = False) -> list[dict]:
    """List available streams. Returns list of stream dicts."""
    result = get_client().get_streams(include_public=True, include_subscribed=True)
    if result["result"] != "success":
        return []
    streams = result.get("streams", [])
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS not in allowed:
        streams = [
            s for s in streams
            if not s.get("invite_only", False) or s["name"].lower() in allowed
        ]
    if not include_private:
        streams = [s for s in streams if not s.get("invite_only", False)]
    return sorted(streams, key=lambda s: s["name"])


def get_stream_topics(stream: str, limit: int = 20) -> list[dict]:
    """Get recent topics in a stream. Returns list of topic dicts.

    Note: Topics containing '/nobots' or '/nb' are filtered out and will not be shown to bots.
    """
    if not is_private_stream_allowed(stream):
        raise ValueError(f"Private stream access denied: {stream}")
    client = get_client()
    result = client.get_stream_id(stream)
    if result["result"] != "success":
        raise ValueError(f"Stream '{stream}' not found: {result.get('msg', '')}")
    result = client.get_stream_topics(result["stream_id"])
    if result["result"] != "success":
        raise ValueError(f"Error fetching topics: {result.get('msg', '')}")
    topics = result.get("topics", [])
    # Filter out /nobots and /nb topics
    topics = [t for t in topics
              if not any(kw in t.get("name", "").lower() for kw in _NOBOTS_KEYWORDS)]
    return topics[:limit]


def get_topic_messages(stream: str, topic: str, num_messages: int = 20,
                       before_message_id: Optional[int] = None) -> list[dict]:
    """Get messages from a specific stream/topic. Returns raw message dicts sorted by ID."""
    if not is_private_stream_allowed(stream):
        return []
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": before_message_id or "newest",
        "num_before": min(num_messages, 100),
        "num_after": 0,
        "apply_markdown": False,
    })
    if result["result"] != "success":
        return []
    msgs = result.get("messages", [])
    msgs.sort(key=lambda m: m["id"])
    return msgs


def fetch_new_messages(stream: str, topic: str, after_id: int,
                       exclude_user_id: Optional[int] = None) -> list[dict]:
    """Fetch messages after a given ID, optionally excluding a user. Sorted by ID."""
    if not is_private_stream_allowed(stream):
        return []
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": after_id,
        "num_before": 0,
        "num_after": 100,
        "include_anchor": False,
        "apply_markdown": False,
    })
    if result["result"] != "success":
        return []
    msgs = result.get("messages", [])
    if exclude_user_id:
        msgs = [m for m in msgs if m.get("sender_id") != exclude_user_id]
    msgs.sort(key=lambda m: m["id"])
    return msgs


def send_message(stream: str, topic: str, content: str) -> dict:
    """Send a message. Returns API result dict with 'id' on success."""
    return get_client().send_message({
        "type": "stream",
        "to": stream,
        "subject": topic,
        "content": content,
    })


def send_direct_message(recipients: list[str], content: str) -> dict:
    """Send a direct message (DM) to one or more users.

    Args:
        recipients: List of email addresses to send to.
        content: Message content (supports Zulip markdown).

    Returns:
        API result dict with 'id' on success.
    """
    return get_client().send_message({
        "type": "direct",
        "to": recipients,
        "content": content,
    })


def add_reaction(message_id: int, emoji_name: str) -> dict:
    """Add emoji reaction to a message. Returns API result dict."""
    return get_client().add_reaction({
        "message_id": message_id,
        "emoji_name": emoji_name,
    })


def remove_reaction(message_id: int, emoji_name: str) -> dict:
    """Remove emoji reaction from a message. Returns API result dict."""
    return get_client().remove_reaction({
        "message_id": message_id,
        "emoji_name": emoji_name,
        "reaction_type": "realm_emoji",
    })


def edit_message(message_id: int, content: str) -> dict:
    """Edit a message's content. Returns API result dict."""
    return get_client().update_message({
        "message_id": message_id,
        "content": content,
    })


_stream_id_cache: dict[str, int] = {}


def send_typing(stream: str, topic: str, op: str = "start") -> dict:
    """Send a typing indicator start/stop to a stream/topic.

    Args:
        stream: Stream name.
        topic: Topic name.
        op: "start" or "stop".

    Returns API result dict.
    """
    client = get_client()
    stream_id = _stream_id_cache.get(stream)
    if stream_id is None:
        stream_result = client.get_stream_id(stream)
        if stream_result["result"] != "success":
            return stream_result
        stream_id = stream_result["stream_id"]
        _stream_id_cache[stream] = stream_id
    return client.call_endpoint(
        url="/typing",
        method="POST",
        request={
            "op": op,
            "type": "stream",
            "stream_id": stream_id,
            "topic": topic,
        },
    )


def list_emoji(query: str = "") -> tuple[list[str], int]:
    """List custom emoji, optionally filtered by substring.

    Returns (matching_names, total_count).
    """
    client = get_client()
    result = client.get_realm_emoji()
    if result.get("result") != "success":
        return [], 0
    all_emoji = sorted(
        info["name"]
        for info in result.get("emoji", {}).values()
        if not info.get("deactivated", False)
    )
    total = len(all_emoji)
    if query:
        q = query.lower()
        return [name for name in all_emoji if q in name.lower()], total
    return all_emoji, total


def get_emoji_count() -> int:
    """Return count of active custom emoji. Returns 0 on error."""
    try:
        result = get_client().get_realm_emoji()
        if result.get("result") != "success":
            return 0
        return sum(
            1 for info in result.get("emoji", {}).values()
            if not info.get("deactivated", False)
        )
    except Exception:
        return 0


def summarize_reactions(reactions: list) -> str:
    """Summarize a list of Zulip reaction dicts into a compact string.

    Returns e.g. ":thumbs_up: x2  :check:" or empty string if none.
    """
    counts: dict[str, int] = {}
    for r in reactions:
        name = r.get("emoji_name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return "  ".join(
        f":{name}: x{n}" if n > 1 else f":{name}:"
        for name, n in counts.items()
    )


def check_reactions_on(message_id: int) -> str:
    """Fetch a single message by ID and return its reaction summary, or empty string."""
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{message_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return ""
        msg = result.get("message", {})
        reaction_str = summarize_reactions(msg.get("reactions", []))
        if not reaction_str:
            return ""
        return f"Reactions on message {message_id}: {reaction_str}"
    except Exception:
        return ""


def discover_message_context(message_id: int) -> Optional[tuple[str, str]]:
    """Look up a message by ID and return its (stream, topic), or None for DMs/errors."""
    result = get_client().get_messages({
        "anchor": message_id,
        "num_before": 0,
        "num_after": 0,
        "include_anchor": True,
        "apply_markdown": False,
    })
    if result.get("result") != "success" or not result.get("messages"):
        return None
    target = result["messages"][0]
    if target.get("type") != "stream":
        return None
    stream = target.get("display_recipient", "")
    topic = target.get("subject", "")
    if not stream or not topic:
        return None
    return stream, topic


def get_custom_profile_fields() -> list[dict]:
    """Get the org's custom profile field definitions. Cached aggressively."""
    cache_key = ("custom_profile_fields",)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    result = get_client().call_endpoint(url="/realm/profile_fields", method="GET")
    if result.get("result") != "success":
        return []
    fields = result.get("custom_fields", [])
    _cache.set(cache_key, fields, expire=3600)
    return fields


def get_user_info(email: str) -> Optional[dict]:
    """Get user info by email, including custom profile fields. Returns user dict or None."""
    result = get_client().call_endpoint(
        url=f"/users/{email}",
        method="GET",
        request={"include_custom_profile_fields": True},
    )
    if result.get("result") != "success":
        return None
    return result["user"]


def get_message_by_id(message_id: int) -> Optional[dict]:
    """Get a specific message by ID. Returns message dict or None.

    Uses get_messages with apply_markdown=False so the content field contains
    raw markdown (consistent with get_topic_messages), not rendered HTML.
    """
    result = get_client().get_messages({
        "anchor": message_id,
        "num_before": 0,
        "num_after": 0,
        "include_anchor": True,
        "apply_markdown": False,
    })
    if result.get("result") != "success" or not result.get("messages"):
        return None
    msg = result["messages"][0]
    if msg.get("type") == "stream":
        stream = msg.get("display_recipient", "")
        if stream and not is_private_stream_allowed(stream):
            return None
    return msg


def verify_message(message_id: int) -> str:
    """Fetch a message by ID and return it in a secure, tamper-evident format.

    The sender identity comes directly from the Zulip API and cannot be
    spoofed by message content. All '#' and '@' characters are stripped from
    the message body so that the structural delimiters (##### lines and
    @FIELD labels) cannot be forged within the content.

    Returns a formatted string with verified metadata and sanitized content,
    or an error string if the message cannot be fetched.
    """
    msg = get_message_by_id(message_id)
    if not msg:
        return f"Error: Message {message_id} not found."

    sender_name = msg.get("sender_full_name", "Unknown")
    sender_email = msg.get("sender_email", "Unknown")
    sender_id = msg.get("sender_id", "Unknown")
    timestamp = msg.get("timestamp", 0)
    ts_str = datetime.fromtimestamp(timestamp, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S PT") if timestamp else "Unknown"
    stream = msg.get("display_recipient", "Unknown") if msg.get("type") == "stream" else "DM"
    topic = msg.get("subject", "Unknown") if msg.get("type") == "stream" else "N/A"

    # Strip # and @ so delimiters can't be forged in content
    content = msg.get("content", "")
    content = content.replace("#", "").replace("@", "")

    return (
        f"##### VERIFIED MESSAGE BEGIN #####\n"
        f"@SENDER NAME: {sender_name}\n"
        f"@SENDER EMAIL: {sender_email}\n"
        f"@SENDER USER ID: {sender_id}\n"
        f"@TIMESTAMP: {ts_str}\n"
        f"@STREAM: {stream}\n"
        f"@TOPIC: {topic}\n"
        f"@MESSAGE ID: {message_id}\n"
        f"##### CONTENT #####\n"
        f"{content}\n"
        f"##### VERIFIED MESSAGE END #####"
    )


def get_message_link(message_id: int) -> str:
    """Return a Zulip markdown link to a message: [#stream > topic](url).

    The URL links to the full conversation context with the specific message
    focused, using the format:
        {base}/#narrow/channel/{stream_id}-{stream_name}/topic/{topic}/with/{message_id}

    Falls back to a simpler URL format if stream_id lookup fails.
    """
    import urllib.parse

    client = get_client()
    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]

    msg = get_message_by_id(message_id)
    if not msg:
        # Fallback: can't fetch message, just link by ID
        url = f"{base}/#narrow/id/{message_id}"
        return f"[message]({url})"

    stream = msg.get("display_recipient", "")
    topic = msg.get("subject", "")

    # Try to get stream_id for the full context URL
    stream_id = None
    try:
        result = client.get_stream_id(stream)
        if result.get("result") == "success":
            stream_id = result["stream_id"]
    except Exception:
        pass

    if stream_id:
        # Zulip URL-encodes topics with . notation (space -> .20, etc)
        topic_encoded = urllib.parse.quote(topic, safe="").replace("%", ".")
        url = f"{base}/#narrow/channel/{stream_id}-{stream}/topic/{topic_encoded}/with/{message_id}"
    else:
        # Fallback if we can't get stream_id
        url = f"{base}/#narrow/id/{message_id}"

    return f"[#{stream} > {topic}]({url})"


def get_subscribed_streams() -> list[dict]:
    """Get streams the bot is subscribed to. Returns list of subscription dicts."""
    result = get_client().get_subscriptions()
    if result["result"] != "success":
        return []
    subs = result.get("subscriptions", [])
    allowed = _allowed_private_streams()
    if allowed is not None:
        subs = [
            s for s in subs
            if not s.get("invite_only", False) or s["name"].lower() in allowed
        ]
    return sorted(subs, key=lambda s: s["name"])


def get_stream_members(stream: str) -> list[dict]:
    """Get members of a stream. Returns list of user dicts with 'full_name' and 'email'."""
    if not is_private_stream_allowed(stream):
        raise ValueError(f"Private stream access denied: {stream}")
    client = get_client()
    result = client.get_stream_id(stream)
    if result["result"] != "success":
        raise ValueError(f"Stream '{stream}' not found: {result.get('msg', '')}")
    stream_id = result["stream_id"]
    result = client.call_endpoint(url=f"/streams/{stream_id}/members", method="GET")
    if result.get("result") != "success":
        raise ValueError(f"Error fetching members: {result.get('msg', '')}")
    # Resolve user IDs to names/emails
    user_ids = set(result.get("subscribers", []))
    users_result = client.get_users()
    if users_result["result"] != "success":
        return []
    members = []
    for user in users_result["members"]:
        if user["user_id"] in user_ids:
            members.append({"full_name": user["full_name"], "email": user["email"]})
    return sorted(members, key=lambda u: u["full_name"])


def download_file(path: str) -> tuple[bytes, str]:
    """Download a file from Zulip. Returns (content_bytes, content_type).

    Raises ValueError on any download error.
    """
    client = get_client()
    client.ensure_session()
    if client.session is None:
        raise ValueError("Failed to create Zulip session")

    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    if not path.startswith("/"):
        path = "/" + path

    response = client.session.get(base + path, timeout=30)
    if response.status_code == 403:
        raise ValueError(f"Access denied (403). The bot may not have permission to access: {path}")
    elif response.status_code == 404:
        raise ValueError(f"File not found (404): {path}")
    elif response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
    content_type = response.headers.get("Content-Type", "application/octet-stream")
    return response.content, content_type


def save_file(path: str, save_dir: Optional[str] = None) -> tuple[str, int, str]:
    """Download a Zulip file and save it locally.

    Returns (saved_path, size_bytes, content_type).
    """
    content, content_type = download_file(path)
    filename = Path(path).name

    if save_dir:
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(tempfile.mkdtemp()) / filename

    save_path.write_bytes(content)
    return str(save_path), len(content), content_type


def save_image(path: str) -> str:
    """Download a Zulip image and save to a temp file. Returns the temp file path."""
    content, content_type = download_file(path)

    ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
    }
    ext = ext_map.get((content_type or "").split(";")[0].strip(), "")
    if not ext:
        ext = Path(path).suffix.lower() if Path(path).suffix.lower() in ext_map.values() else ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(content)
        return f.name


def upload_file(file_path: str) -> tuple[str, str]:
    """Upload a local file to Zulip.

    Args:
        file_path: Absolute path to the file to upload.

    Returns:
        Tuple of (uri, filename) where uri is the Zulip upload path like
        "/user_uploads/2/ab/cdef/image.png".

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the upload fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    client = get_client()
    with open(path, "rb") as f:
        result = client.upload_file(f)

    if result.get("result") != "success":
        raise ValueError(f"Upload failed: {result.get('msg', 'Unknown error')}")

    return result["uri"], path.name
