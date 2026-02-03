import time
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

import diskcache
import zulip

TIMEZONE = ZoneInfo("America/Los_Angeles")

_client: Optional[zulip.Client] = None
_cache = diskcache.Cache(Path(__file__).parent.parent / ".cache")
_ignored_streams: set[str] = set()


def set_ignored_streams(streams: set[str]) -> None:
    """Set streams to exclude from all message fetches."""
    global _ignored_streams
    _ignored_streams = {s.lower() for s in streams}


def get_ignored_streams() -> set[str]:
    """Return the current ignored streams set."""
    return _ignored_streams


def get_client() -> zulip.Client:
    global _client
    if _client is None:
        import os
        rc_env = os.environ.get("ZULIP_RC_PATH")
        if rc_env:
            config_path = Path(rc_env)
        else:
            config_path = Path(__file__).parent.parent / ".zuliprc"
        assert config_path.exists(), f"zuliprc not found at: {config_path}"
        _client = zulip.Client(config_file=str(config_path))
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


class _ZulipHTMLParser(HTMLParser):
    """Converts Zulip HTML to clean plaintext via tree walk.

    Handles the common cases cleanly (paragraphs, line breaks, links, code,
    blockquotes, spoilers, lists, images). Anything exotic passes through as-is.
    """

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._stack: list[str] = []  # tag stack
        self._attrs_stack: list[dict] = []
        self._suppress = False  # inside a tag whose text we skip

    def _in(self, cls_name: str) -> bool:
        """Check if we're inside a div with the given class."""
        return any(a.get("class", "") and cls_name in a.get("class", "")
                   for a in self._attrs_stack)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self._stack.append(tag)
        self._attrs_stack.append(a)

        if tag == "br":
            self._parts.append("\n")
            self._stack.pop()
            self._attrs_stack.pop()
        elif tag == "p":
            if not self._in("spoiler-header"):
                if self._parts and not self._parts[-1].endswith("\n\n"):
                    self._parts.append("\n\n")
        elif tag == "blockquote":
            self._parts.append("\n<BLOCKQUOTE>")
        elif tag == "code":
            # Check if inside a <pre> (code block) or standalone (inline)
            if "pre" in self._stack[:-1]:
                self._parts.append("\n```\n")
            else:
                self._parts.append("`")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "img":
            src = a.get("src", "")
            if src:
                self._parts.append(f"[image: {src}]")
            self._stack.pop()
            self._attrs_stack.pop()
        elif tag == "div":
            cls = a.get("class", "")
            if "spoiler-header" in cls:
                self._parts.append("\n<spoiler title=\"")
            elif "spoiler-content" in cls:
                self._parts.append("\n")
        elif tag == "span":
            cls = a.get("class", "")
            if "katex" in cls:
                self._suppress = True
        elif tag == "annotation":
            # Inside katex — this has the raw TeX
            if a.get("encoding") == "application/x-tex":
                self._suppress = False
                self._parts.append("$")

    def handle_endtag(self, tag):
        if tag == "br" or tag == "img":
            return

        if tag == "blockquote":
            self._parts.append("</BLOCKQUOTE>\n")
        elif tag == "code":
            if "pre" in self._stack[:-1]:
                self._parts.append("\n```\n")
            else:
                self._parts.append("`")
        elif tag == "a":
            pass  # link text emitted by handle_data
        elif tag == "div":
            a = self._attrs_stack[-1] if self._attrs_stack else {}
            cls = a.get("class", "")
            if "spoiler-header" in cls:
                self._parts.append("\">\n")
            elif "spoiler-block" in cls:
                self._parts.append("\n</spoiler>\n")
        elif tag == "annotation":
            self._parts.append("$")
            self._suppress = True  # suppress rest of katex
        elif tag == "span":
            a = self._attrs_stack[-1] if self._attrs_stack else {}
            if "katex" in a.get("class", ""):
                self._suppress = False

        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
            self._attrs_stack.pop()

    def handle_data(self, data):
        if self._suppress:
            return
        if self._in("spoiler-header"):
            data = data.strip()
            if not data:
                return
        self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Process blockquotes (we used markers to avoid nesting issues)
        while "<BLOCKQUOTE>" in text:
            def _quote_block(segment):
                lines = segment.split("\n")
                return "\n".join(f"> {l}" if l else ">" for l in lines)
            start = text.index("<BLOCKQUOTE>")
            end = text.index("</BLOCKQUOTE>", start)
            inner = text[start + len("<BLOCKQUOTE>"):end]
            quoted = _quote_block(inner.strip())
            text = text[:start] + "\n" + quoted + "\n" + text[end + len("</BLOCKQUOTE>"):]
        # Collapse excess newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert Zulip HTML content to clean plaintext."""
    parser = _ZulipHTMLParser()
    parser.feed(html)
    return parser.get_text()


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
    """
    if not messages:
        return ""

    if combine:
        messages = _combine_messages(messages)

    lines = []
    prev_timestamp = None
    for msg in messages:
        sender = msg.get("sender_full_name", "Unknown")
        msg_id = msg.get("id", "")
        content = _html_to_text(msg.get("content", ""))
        timestamp = msg.get("timestamp", 0)

        # Build the time attribute
        time_attr = _time_attr(msg, prev_timestamp)

        attrs = [f'user="{sender}"']
        if time_attr:
            attrs.append(time_attr)
        if include_topic and msg.get("type") == "stream":
            attrs.append(f'stream="{msg.get("display_recipient", "")}"')
            attrs.append(f'topic="{msg.get("subject", "")}"')
        attrs.append(f'id="{msg_id}"')

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

    for msg in messages[1:]:
        same_user = msg.get("sender_full_name") == current.get("sender_full_name")
        time_gap = msg.get("timestamp", 0) - current.get("time_range_end", current.get("timestamp", 0))

        if same_user and time_gap <= 120:
            # Merge: append content, update end time and id
            current["content"] = current["content"] + "\n\n" + msg.get("content", "")
            current["time_range_end"] = msg.get("timestamp", 0)
            current["id"] = msg.get("id", current["id"])
        else:
            combined.append(current)
            current = dict(msg)

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
        header = f"Stream: {stream}, Topic: {topic}"
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
    if not include_private:
        streams = [s for s in streams if not s.get("invite_only", False)]
    return sorted(streams, key=lambda s: s["name"])


def get_stream_topics(stream: str, limit: int = 20) -> list[dict]:
    """Get recent topics in a stream. Returns list of topic dicts."""
    client = get_client()
    result = client.get_stream_id(stream)
    if result["result"] != "success":
        raise ValueError(f"Stream '{stream}' not found: {result.get('msg', '')}")
    result = client.get_stream_topics(result["stream_id"])
    if result["result"] != "success":
        raise ValueError(f"Error fetching topics: {result.get('msg', '')}")
    return result.get("topics", [])[:limit]


def get_topic_messages(stream: str, topic: str, num_messages: int = 20,
                       before_message_id: Optional[int] = None) -> list[dict]:
    """Get messages from a specific stream/topic. Returns raw message dicts sorted by ID."""
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": before_message_id or "newest",
        "num_before": min(num_messages, 100),
        "num_after": 0,
    })
    if result["result"] != "success":
        return []
    msgs = result.get("messages", [])
    msgs.sort(key=lambda m: m["id"])
    return msgs


def fetch_new_messages(stream: str, topic: str, after_id: int,
                       exclude_user_id: Optional[int] = None) -> list[dict]:
    """Fetch messages after a given ID, optionally excluding a user. Sorted by ID."""
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": after_id,
        "num_before": 0,
        "num_after": 100,
        "include_anchor": False,
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


def get_user_info(email: str) -> Optional[dict]:
    """Get user info by email. Returns user dict or None."""
    result = get_client().call_endpoint(url=f"/users/{email}", method="GET")
    if result.get("result") != "success":
        return None
    return result["user"]


def get_message_by_id(message_id: int) -> Optional[dict]:
    """Get a specific message by ID. Returns message dict or None."""
    result = get_client().get_raw_message(message_id)
    if result.get("result") != "success":
        return None
    return result["message"]


def get_message_link(message_id: int) -> str:
    """Return a Zulip markdown link to a message: [#stream > topic](url).

    Falls back to a bare URL if the message can't be fetched.
    """
    client = get_client()
    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    url = f"{base}/#narrow/id/{message_id}"

    msg = get_message_by_id(message_id)
    if msg:
        stream = msg.get("display_recipient", "")
        topic = msg.get("subject", "")
        return f"[#{stream} > {topic}]({url})"
    return f"[message]({url})"


def get_subscribed_streams() -> list[dict]:
    """Get streams the bot is subscribed to. Returns list of subscription dicts."""
    result = get_client().get_subscriptions()
    if result["result"] != "success":
        return []
    return sorted(result.get("subscriptions", []), key=lambda s: s["name"])


def get_stream_members(stream: str) -> list[dict]:
    """Get members of a stream. Returns list of user dicts with 'full_name' and 'email'."""
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
    if response.status_code != 200:
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
