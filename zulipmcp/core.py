import functools
import hashlib
import os
import re
import tempfile
import json
import math
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Callable, NamedTuple, Optional

import diskcache
import requests
import zulip
from requests.adapters import HTTPAdapter

_DEFAULT_TIMEZONE = "America/Los_Angeles"


def _load_timezone() -> ZoneInfo:
    """Load the configured IANA timezone."""
    name = os.environ.get("ZULIPMCP_TIMEZONE", _DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Invalid ZULIPMCP_TIMEZONE {name!r}; use an IANA timezone name"
        ) from exc


TIMEZONE = _load_timezone()

_client: Optional[zulip.Client] = None
_ignored_streams: set[str] = set()
_ALL_PRIVATE_STREAMS = "__ALL__"
_dismiss_emoji: set[str] = {"stop_sign"}

# Zulip silently truncates messages longer than this server-side, so a sender
# never learns its message was clipped. Matches Zulip's default; override via
# env for realms configured with a different cap. Bad values fall back rather
# than crashing the server at import.
try:
    MAX_MESSAGE_LENGTH = int(os.environ.get("ZULIP_MAX_MESSAGE_LENGTH", "10000"))
except ValueError:
    MAX_MESSAGE_LENGTH = 10000

_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')
# Inline code spans with variable-length delimiters (`` `x` ``, ``` ``x`` ```),
# mirroring Python-Markdown's backtick pairing.
_INLINE_CODE_RE = re.compile(r'(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)')
# Link URLs allow one level of balanced parens (GFM-style) so targets like
# .../wiki/Foo_(bar) terminate at the real closing paren.
_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\((<?(?:\([^()\s]*\)|[^()\s])+>?)\)')
_URL_ABUTS_BOLD_RE = re.compile(r'(https?://[^\s<>\[\]()*]+)(?=\*\*)')


def _autofix_enabled() -> bool:
    """Whether outgoing markdown normalization is enabled.

    Controlled by ``ZULIPMCP_MARKDOWN_AUTOFIX``: set to ``0`` or ``false``
    (case-insensitive) to disable all normalization.  Read at call time so
    it can be toggled without a restart.
    """
    return os.environ.get('ZULIPMCP_MARKDOWN_AUTOFIX', '').lower() not in ('0', 'false')


def _is_table_separator(line: str) -> bool:
    """Whether *line* is a markdown table separator (e.g. ``| --- | --- |``)."""
    stripped = line.strip()
    if not stripped or '-' not in stripped:
        return False
    inner = stripped.lstrip('|').rstrip('|')
    return bool(inner) and set(inner) <= set('|:- ')


def _restyle_bold_in_link(m: 're.Match[str]') -> str:
    """Rewrite bold inside a markdown link's text so Zulip renders it.

    Whole-text bold moves outside the link (``[**a**](url)`` becomes
    ``**[a](url)**``); partial bold is stripped from the text.
    """
    text, url = m.group(1), m.group(2)
    if '**' not in text:
        return m.group(0)
    if text.startswith('**') and text.endswith('**') and text.count('**') == 2:
        return f'**[{text[2:-2]}]({url})**'
    return f'[{text.replace("**", "")}]({url})'


def _sub_outside_code(pattern: 're.Pattern[str]',
                      repl: Callable[['re.Match[str]'], str], line: str) -> str:
    """Apply ``pattern.sub(repl, line)``, leaving matches inside inline code alone.

    A match is exempt only when wholly enclosed in an inline code span; a
    match that merely *contains* a code span is still rewritten.
    """
    code_spans = [span.span() for span in _INLINE_CODE_RE.finditer(line)]

    def guarded(m: 're.Match[str]') -> str:
        if any(s <= m.start() and m.end() <= e for s, e in code_spans):
            return m.group(0)
        return repl(m)

    return pattern.sub(guarded, line)


def _fix_bold_links(line: str) -> str:
    """Rewrite bold/link combinations that Zulip's renderer breaks on."""
    if '**' not in line:
        return line
    line = _sub_outside_code(_LINK_RE, _restyle_bold_in_link, line)
    line = _sub_outside_code(_URL_ABUTS_BOLD_RE,
                             lambda m: f'[{m.group(1)}]({m.group(1)})', line)
    return line


def normalize_zulip_markdown(content: str) -> str:
    """Normalize outgoing markdown for Zulip's renderer.

    Two deterministic fixes for patterns LLMs trained on GFM emit constantly:

    1. Blank lines before tables.  Zulip's Python-Markdown parser requires a
       blank line before table header rows; injects the missing one.
    2. Bold/link combos Zulip breaks on.  Bold inside link text
       (``[**a**](url)``) renders as literal escaped asterisks with an
       unclickable link (zulip/zulip#36087, fix PR stalled upstream) —
       whole-text bold moves outside the link, partial bold is stripped.
       A bare URL immediately followed by ``**`` is wrapped as
       ``[url](url)``: a trailing ``**`` otherwise gets swallowed into the
       autolinked URL, and a URL *preceded* by ``**`` never autolinks at
       all, so the rewrite also upgrades bold URLs to clickable links.

    Both fixes skip fenced code blocks and indented code.  The bold/link
    fix also skips inline code spans; the table fix also skips blockquotes.
    Accepted edge case: a URL whose path literally contains ``**`` is
    truncated at the asterisks (literal ``*`` in URLs should be
    percent-encoded).

    Set ``ZULIPMCP_MARKDOWN_AUTOFIX=0`` (or ``false``) to disable all
    normalization and send content verbatim.
    """
    if not _autofix_enabled():
        return content

    lines = content.split('\n')
    result: list[str] = []
    in_fence = False
    fence_char: str | None = None
    fence_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        m = _FENCE_RE.match(stripped)
        if m:
            fence_chars = m.group(1)
            after_fence = stripped[len(fence_chars):]
            if not in_fence:
                in_fence, fence_char, fence_len = True, fence_chars[0], len(fence_chars)
            elif fence_chars[0] == fence_char and len(fence_chars) >= fence_len and not after_fence:
                in_fence = False
                fence_char = None
            result.append(line)
            continue

        if in_fence:
            result.append(line)
            continue

        if (
            stripped
            and '|' in stripped
            and result
            and result[-1].strip()
            and not stripped.startswith('>')
            and not line.startswith('    ')
            and not line.startswith('\t')
            and i + 1 < len(lines)
            and _is_table_separator(lines[i + 1])
        ):
            result.append('')

        if not line.startswith('    ') and not line.startswith('\t'):
            line = _fix_bold_links(line)

        result.append(line)

    return '\n'.join(result)


def _is_allow_all_token(value: object) -> bool:
    """Whether a value is the special allow-all sentinel."""
    return isinstance(value, str) and value.strip().upper() == _ALL_PRIVATE_STREAMS


def _parse_stream_allowlist(env_var: str, default_all: bool) -> set[str]:
    """Parse a stream allowlist env var into a lowercase set."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return {_ALL_PRIVATE_STREAMS} if default_all else set()
    if _is_allow_all_token(raw):
        return {_ALL_PRIVATE_STREAMS}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            if any(_is_allow_all_token(s) for s in parsed):
                return {_ALL_PRIVATE_STREAMS}
            return {str(s).lower() for s in parsed if str(s).strip()}
        if isinstance(parsed, str):
            if _is_allow_all_token(parsed):
                return {_ALL_PRIVATE_STREAMS}
            return {parsed.lower()}
    except json.JSONDecodeError:
        return {s.strip().lower() for s in raw.split(",") if s.strip()}
    return set()


def _allowed_private_streams() -> set[str]:
    """Return allowed private streams from env.

    Security model:
    - Unset/empty env => no private stream access (default-deny)
    - "__ALL__" => explicit full private stream access
    - list/string => explicit allowlist (normalized to lowercase)
    """
    return _parse_stream_allowlist("BOT_ALLOWED_PRIVATE_STREAMS", default_all=False)


def _allowed_write_streams() -> set[str]:
    """Return allowed write streams from env.

    Security model:
    - Unset/empty env => all stream writes allowed (backwards-compatible)
    - "__ALL__" => explicit full write access
    - list/string => explicit allowlist (normalized to lowercase)
    """
    return _parse_stream_allowlist("BOT_ALLOWED_WRITE_STREAMS", default_all=True)


def is_private_stream_allowed(stream_name: str) -> bool:
    """Whether the current process is allowed to access this private stream."""
    if not is_stream_private(stream_name):
        return True
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS in allowed:
        return True
    return stream_name.lower() in allowed


def dm_reads_allowed() -> bool:
    """Whether DMs may be read: only under ``BOT_ALLOWED_PRIVATE_STREAMS=__ALL__``.

    An explicit stream list never unlocks DMs, which on a shared bot belong
    to every user who has messaged it.
    """
    return _ALL_PRIVATE_STREAMS in _allowed_private_streams()


def allowlisted_private_streams() -> list[str]:
    """Sorted names in an explicit ``BOT_ALLOWED_PRIVATE_STREAMS`` list; empty when unset or ``__ALL__``."""
    allowed = _allowed_private_streams()
    return [] if _ALL_PRIVATE_STREAMS in allowed else sorted(allowed)


PRIVATE_STREAM_DENIED = "Private stream access denied: not in BOT_ALLOWED_PRIVATE_STREAMS."
DM_READS_DENIED = "Reading DMs requires BOT_ALLOWED_PRIVATE_STREAMS=__ALL__."


def is_stream_write_allowed(stream_name: str) -> bool:
    """Whether the current process is allowed to send to this stream."""
    allowed = _allowed_write_streams()
    if _ALL_PRIVATE_STREAMS in allowed:
        return True
    return stream_name.lower() in allowed


def _stream_write_error(stream: str) -> Optional[dict]:
    """Return an API-style error dict if writes to this stream are forbidden."""
    if not is_private_stream_allowed(stream):
        return {"result": "error", "msg": f"Private stream access denied: {stream}"}
    if not is_stream_write_allowed(stream):
        return {"result": "error", "msg": f"Stream write access denied: {stream}"}
    return None


def _get_stream_message_for_write(message_id: int) -> tuple[Optional[dict], Optional[dict]]:
    """Return a stream message or an API-style error dict for write operations."""
    msg = get_message_by_id(message_id)
    if not msg:
        return None, {"result": "error", "msg": f"Message {message_id} not found or inaccessible"}
    if msg.get("type") != "stream":
        return None, {"result": "error", "msg": "This operation only supports stream messages"}
    stream = msg.get("display_recipient", "")
    if not stream:
        return None, {"result": "error", "msg": f"Message {message_id} stream not found"}
    err = _stream_write_error(stream)
    if err:
        return None, err
    return msg, None


def set_ignored_streams(streams: set[str]) -> None:
    """Set streams to exclude from message fetches; fetch_narrow keeps them when its narrow names a channel."""
    global _ignored_streams
    _ignored_streams = {s.lower() for s in streams}


def get_ignored_streams() -> set[str]:
    """Return the current ignored streams set."""
    return _ignored_streams


def set_dismiss_emoji(emoji: set[str] | str) -> None:
    """Set the emoji names that trigger session dismissal when reacted.

    Pass an empty set to disable dismiss-by-reaction.
    """
    global _dismiss_emoji
    if isinstance(emoji, str):
        emoji = {emoji}
    _dismiss_emoji = {e.strip(":") for e in emoji if e.strip(":")}


def get_dismiss_emoji() -> set[str]:
    """Return the current dismiss emoji set."""
    return _dismiss_emoji


def is_dismiss_reaction(event: dict, bot_user_id: int,
                        stream: Optional[str] = None,
                        topic: Optional[str] = None) -> bool:
    """Check if a reaction event is a dismiss signal on a bot-authored message.

    Returns True if:
    - The reaction is an "add" operation
    - The emoji matches a dismiss emoji
    - The reactor is not the bot itself
    - The reacted-on message was sent by the bot
    - The reacted-on message is in the given stream/topic (if specified)

    The last checks require one API call to fetch the message.

    Note: Zulip's event queue narrow does NOT filter reaction events, so
    the caller receives reactions from all streams.  Pass stream/topic to
    ensure only reactions in the current conversation trigger dismissal.
    """
    if event.get("op") != "add":
        return False
    if event.get("emoji_name") not in _dismiss_emoji:
        return False
    if event.get("user_id") == bot_user_id:
        return False

    # Verify the reacted-on message was sent by this bot (and optionally
    # that it belongs to the expected stream/topic).
    msg_id = event.get("message_id")
    if not msg_id:
        return False
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{msg_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return False
        msg = result.get("message", {})
        if msg.get("sender_id") != bot_user_id:
            return False
        # Verify stream/topic if provided.
        # display_recipient is a string for stream messages but a list for DMs.
        display_recipient = msg.get("display_recipient", "")
        if stream:
            if not isinstance(display_recipient, str) or display_recipient.lower() != stream.lower():
                return False
        if topic and msg.get("subject", "").lower() != topic.lower():
            return False
        return True
    except Exception:
        return False


def check_dismissed(message_id: int, bot_user_id: int) -> Optional[str]:
    """Check if a bot message already has a dismiss reaction.

    Fetches the message via REST API and inspects its reactions list.
    Returns the emoji name if a dismiss reaction from a non-bot user exists,
    or None if no dismiss reaction is found.

    This catches the race condition where a user reacts while the bot is
    busy (not in listen()), so the reaction event was never consumed.
    """
    if not _dismiss_emoji:
        return None
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{message_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return None
        msg = result.get("message", {})
        for reaction in msg.get("reactions", []):
            if reaction.get("emoji_name") in _dismiss_emoji and reaction.get("user_id") != bot_user_id:
                return reaction["emoji_name"]
        return None
    except Exception:
        return None


def is_stream_private(stream_name: str) -> bool:
    """Check if a stream is private (invite_only). Cached for 1 hour."""
    cache_key = ("is_stream_private", stream_name.lower())
    cached = _get_cache().get(cache_key)
    if cached is not None:
        return cached

    # Fetch all streams and cache every one — avoids repeated API calls
    result = get_client().get_streams(include_public=True, include_subscribed=True)
    if result["result"] != "success":
        return True  # can't determine — assume private (safer)

    for s in result.get("streams", []):
        is_private = s.get("invite_only", False)
        _get_cache().set(("is_stream_private", s["name"].lower()), is_private, expire=3600)

    # Re-check cache after populating
    cached = _get_cache().get(cache_key)
    return cached if cached is not None else True  # not in results — must be private


@functools.cache
def _open_cache(directory: Path) -> diskcache.Cache:
    return diskcache.Cache(directory)


def _get_cache() -> diskcache.Cache:
    """Disk cache scoped to the current client's site and account.

    Cache keys carry no identity, so processes authenticated as different
    users must not share a directory.
    """
    client = get_client()
    site = getattr(client, "base_url", "")
    identity = hashlib.sha256(f"{site}|{client.email}".encode()).hexdigest()[:16]
    base = os.environ.get("ZULIPMCP_CACHE_DIR") or Path(tempfile.gettempdir()) / "zulipmcp_cache"
    return _open_cache(Path(base) / identity)


class ReadOnlyError(PermissionError):
    """Raised when a write is attempted while ``ZULIPMCP_READ_ONLY`` is on."""


def is_read_only() -> bool:
    """Whether ``ZULIPMCP_READ_ONLY`` is on. Raises ValueError on unrecognized values."""
    raw = os.environ.get("ZULIPMCP_READ_ONLY", "")
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(
        f"Invalid ZULIPMCP_READ_ONLY={raw!r}; use one of 1, true, yes, on, 0, false, no, off, "
        f"or leave it unset"
    )


class _ReadOnlyAdapter(HTTPAdapter):
    """Refuses every non-GET at the transport, beneath API calls, raw ``session.*`` and redirects."""

    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        if request.method != "GET":
            raise ReadOnlyError(f"ZULIPMCP_READ_ONLY is set; refusing {request.method} {request.url}")
        return super().send(request, *args, **kwargs)


class _ReadOnlyClient(zulip.Client):
    """Mounts :class:`_ReadOnlyAdapter` on both schemes so every session request passes through it."""

    def ensure_session(self) -> None:
        if self.session:
            return
        super().ensure_session()
        assert self.session is not None
        for prefix in ("https://", "http://"):
            self.session.mount(prefix, _ReadOnlyAdapter())


def get_client() -> zulip.Client:
    """Get or create the Zulip client singleton.

    Config resolution order:
        1. ZULIP_RC_PATH environment variable (absolute path to zuliprc)
        2. .zuliprc in the current working directory

    With ``ZULIPMCP_READ_ONLY`` on, the client raises :class:`ReadOnlyError`
    for every non-GET request.
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
        client_class = _ReadOnlyClient if is_read_only() else zulip.Client
        client = client_class(config_file=str(config_path))
        if isinstance(client, _ReadOnlyClient):
            # Fail closed if a zulip release builds its session in a way
            # that skips ensure_session() and so the guard.
            client.ensure_session()
            session = client.session
            if session is None or not isinstance(session.get_adapter(client.base_url), _ReadOnlyAdapter):
                raise RuntimeError("ZULIPMCP_READ_ONLY is set but the client session is unguarded")
        _client = client
        # Log identity so it's visible in MCP startup output
        email = _client.email or "unknown"
        mode = " [read-only]" if isinstance(_client, _ReadOnlyClient) else ""
        print(f"[zulipmcp] Zulip client initialized as: {email} (from {config_path}){mode}", file=sys.stderr)
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
    - Uses the configured timezone
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
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif show_time:
        return dt.strftime("%H:%M:%S %Z")
    else:
        return ""


def _time_attr(msg: dict, prev_timestamp: Optional[int]) -> str:
    """Build the time attribute string for a <msg> tag."""
    timestamp = msg.get("timestamp", 0)
    if "time_range_end" in msg:
        start = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        end = datetime.fromtimestamp(msg["time_range_end"], tz=TIMEZONE)
        if start.date() != end.date():
            return f'time="{start.strftime("%Y-%m-%d %H:%M:%S %Z")}-{end.strftime("%Y-%m-%d %H:%M:%S %Z")}"'
        return f'time="{start.strftime("%Y-%m-%d %H:%M:%S %Z")}-{end.strftime("%H:%M:%S %Z")}"'
    ts_str = _format_timestamp(timestamp, prev_timestamp)
    if ts_str:
        return f'time="{ts_str}"'
    return ""


_MENTION_FLAGS = frozenset({
    "mentioned", "stream_wildcard_mentioned", "topic_wildcard_mentioned",
    "group_mentioned", "wildcard_mentioned",
})

_account: Optional[tuple[zulip.Client, Optional[int]]] = None  # (client, user_id if human)


def _human_user_id() -> Optional[int]:
    """This account's user_id if it is a human account, else None.

    Cached per client only on success, so a transient failure (no config,
    network, API error) can't suppress unread markers for the whole process.
    """
    global _account
    try:
        client = get_client()
        if _account is None or _account[0] is not client:
            profile = client.get_profile()
            if profile.get("result") != "success":
                return None
            human = profile.get("is_bot") is False
            _account = (client, profile.get("user_id") if human else None)
    except (OSError, zulip.ZulipError):
        return None
    return _account[1]


# Message text and names are attacker-controlled: an unescaped quote in a
# name adds attributes, and "<msg"/"</msg" in a body forges whole messages.
# Only tag-shaped text is touched (msg then ">" or an attribute), so code like
# "i < msg.length" survives.
_INVISIBLE = (r"[\s\u00ad\u034f\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2069"
              r"\u3164\ufe00-\ufe0f\ufeff]*")  # whitespace, zero-width and bidi format chars
_MSG_TAG_RE = re.compile(
    rf"<(?={_INVISIBLE}/?{_INVISIBLE}m{_INVISIBLE}s{_INVISIBLE}g{_INVISIBLE}(?:/?>|[\w-]+{_INVISIBLE}=))",
    re.IGNORECASE)
# Unicode tag characters render invisibly but models read them as text. Their
# only legitimate use, subdivision flag emoji, degrades to a plain black flag.
_STRIP_TAG_CHARS = dict.fromkeys(range(0xE0000, 0xE0080))
_ATTR_ESCAPES = str.maketrans({'"': "&quot;", "<": "&lt;", ">": "&gt;", **_STRIP_TAG_CHARS})


def escape_msg_tags(text: str) -> str:
    """Neutralize anything in text that reads as a <msg> or </msg> tag; drop tag characters."""
    return _MSG_TAG_RE.sub("&lt;", text.translate(_STRIP_TAG_CHARS))


def _attr(value: Any) -> str:
    """Escape a value for a double-quoted <msg> attribute."""
    return str(value).translate(_ATTR_ESCAPES)


def format_messages(messages: list[dict], include_topic: bool = False,
                    combine: bool = True, max_chars: Optional[int] = None) -> str:
    """Format messages as XML <msg> tags.

    combine=True merges consecutive messages from same user within 2 min.
    Smart timestamps: date shown only when it changes, time shown on 5+ min gaps.
    include_topic=True adds stream and topic attributes to each <msg> tag, or
    a dm attribute listing participant emails for DMs.
    Messages carrying ``flags`` get mentioned="true" when they mention this
    account, and unread="true" when a human account hasn't read them, except
    on the account's own messages.
    max_chars clips each message's content to that many characters, marking
    the cut with the full length.

    Note: Messages in /nobots or /nb topics or starting with /nobots or /nb are automatically
    filtered out and will not be shown to the bot.
    """
    if not messages:
        return ""

    # Filter out messages that should be hidden from bots
    messages = filter_for_bot(messages)
    if not messages:
        return ""

    # Clip before combining so each marker counts its own message.
    if max_chars is not None:
        messages = [_clip_content(m, max_chars) for m in messages]

    # Look up the account only if something could be unread.
    me = (_human_user_id()
          if any("read" not in m.get("flags", ["read"]) for m in messages) else None)

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

        attrs = [f'user="{_attr(sender)}"']
        if time_attr:
            attrs.append(time_attr)
        if include_topic and msg.get("type") == "stream":
            stream_name = msg.get("display_recipient", "")
            visibility = "private" if is_stream_private(stream_name) else "public"
            attrs.append(f'stream="{_attr(stream_name)}"')
            attrs.append(f'visibility="{visibility}"')
            attrs.append(f'topic="{_attr(msg.get("subject", ""))}"')
        elif include_topic and msg.get("type") == "private":
            emails = ", ".join(r.get("email", "") for r in msg.get("display_recipient", []))
            attrs.append(f'dm="{_attr(emails)}"')
        attrs.append(f'id="{_attr(msg_id)}"')

        # Missing flags mean unknown, not unread. API-sent messages start
        # unread for their own sender, so those never count.
        flags = msg.get("flags", ["read"])
        if "read" not in flags and me is not None and msg.get("sender_id") != me:
            attrs.append('unread="true"')
        if _MENTION_FLAGS.intersection(flags):
            attrs.append('mentioned="true"')

        # Add reaction summary as attribute if present
        reaction_str = summarize_reactions(msg.get("reactions", []))
        if reaction_str:
            attrs.append(f'reactions="{_attr(reaction_str)}"')

        tag_open = "<msg " + " ".join(attrs) + ">"
        lines.append(f"{tag_open}\n{escape_msg_tags(content)}\n</msg>")

        prev_timestamp = msg.get("time_range_end", timestamp)

    return "\n".join(lines)


def _clip_content(msg: dict, max_chars: int) -> dict:
    """Return msg with content cut to max_chars plus a marker naming the full length."""
    # Escape before cutting: a tag cut short of its ">" is no longer tag-shaped,
    # so the escape at output would miss the stub.
    content = escape_msg_tags(msg.get("content", ""))
    if len(content) <= max_chars:
        return msg
    marker = f"… [{len(content)} chars; get_message_by_id has the rest]"
    return {**msg, "content": content[:max(max_chars, 0)] + marker}


def _combine_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages from same user within 2 minutes."""
    if not messages:
        return []

    combined = []
    current = dict(messages[0])
    current["reactions"] = list(current.get("reactions", []))

    for msg in messages[1:]:
        same_user = all(msg.get(k) == current.get(k) for k in ("sender_id", "sender_full_name"))
        time_gap = msg.get("timestamp", 0) - current.get("time_range_end", current.get("timestamp", 0))

        if same_user and time_gap <= 120:
            # Merge: append content, update end time, id, and reactions
            current["content"] = current["content"] + "\n\n" + msg.get("content", "")
            current["time_range_end"] = msg.get("timestamp", 0)
            current["id"] = msg.get("id", current["id"])
            current["reactions"].extend(msg.get("reactions", []))
            # A combined block is unread or mentioned if any part is.
            if "flags" in msg:
                prev = current.get("flags", msg["flags"])
                merged = {*prev, *msg["flags"]}
                if "read" not in prev or "read" not in msg["flags"]:
                    merged.discard("read")
                current["flags"] = sorted(merged)
        else:
            combined.append(current)
            current = dict(msg)
            current["reactions"] = list(current.get("reactions", []))

    combined.append(current)
    return combined


# ============================================================================
# Direct API wrappers — return raw Python objects for composability
# ============================================================================

def get_profile() -> dict:
    """Get this account's profile. Returns the profile dict from the API."""
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
    """Get messages from a stream/topic up to and including ``before_message_id``.

    Returns raw message dicts sorted by ID, or [] on error or when the
    stream is a disallowed private stream.
    """
    try:
        page = fetch_narrow(
            build_narrow(stream=stream, topic=topic), num_messages=num_messages,
            before_message_id=before_message_id + 1 if before_message_id else None,
        )
    except ValueError:
        return []
    return page.messages


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


_DATE_ANCHOR_FEATURE_LEVEL = 445


def build_narrow(stream: Optional[str] = None, topic: Optional[str] = None,
                 dm_with: Optional[list[str]] = None, sender: Optional[str] = None,
                 query: Optional[str] = None, channels: Optional[str] = None,
                 only: Optional[list[str]] = None,
                 dm_including: Optional[list[str]] = None) -> list[dict]:
    """Build a Zulip narrow from the given filters; empty filters are omitted.

    ``dm_with`` selects exactly that DM conversation (self is implied).
    ``query`` is Zulip's server-side keyword search. ``channels="public"``
    widens a search to the full history of every public channel.
    ``only`` items become ``is`` terms and ``dm_including`` emails
    ``dm-including`` terms; all terms AND.
    """
    narrow = []
    if channels:
        narrow.append({"operator": "channels", "operand": channels})
    if stream:
        narrow.append({"operator": "channel", "operand": stream})
    if topic:
        narrow.append({"operator": "topic", "operand": topic})
    if dm_with:
        # The server accepts a comma-separated email string or a list of user
        # IDs here; a list of emails is rejected.
        narrow.append({"operator": "dm", "operand": ",".join(dm_with)})
    # The server takes one user per dm-including term.
    for email in dm_including or []:
        narrow.append({"operator": "dm-including", "operand": email})
    if sender:
        narrow.append({"operator": "sender", "operand": sender})
    if query:
        narrow.append({"operator": "search", "operand": query})
    for flag in only or []:
        narrow.append({"operator": "is", "operand": flag})
    return narrow


def _is_readable(msg: dict) -> bool:
    """Whether a fetched message passes the DM and private-stream rules."""
    if msg.get("type") == "private":
        return dm_reads_allowed()
    if msg.get("type") != "stream":
        return True
    return is_private_stream_allowed(msg.get("display_recipient", ""))


class ReadScopeError(ValueError):
    """Raised when a narrow doesn't name a scope this process may read."""


# The server still accepts the underscore spellings of the legacy operators.
_DM_OPERATORS = frozenset({"dm", "dm-including", "pm-with", "pm_with", "group-pm-with", "group_pm_with"})
_CHANNEL_OPERATORS = frozenset({"channel", "stream"})
_CHANNELS_OPERATORS = frozenset({"channels", "streams"})


def _channel_operands(narrow: list[dict]) -> list[str]:
    """Operands of the non-negated ``channel``/``stream`` terms that are names."""
    return [t["operand"] for t in narrow
            if isinstance(t, dict) and t.get("operator") in _CHANNEL_OPERATORS
            and t.get("negated") is not True and isinstance(t.get("operand"), str)]


def _check_read_scope(narrow: list[dict]) -> None:
    """Raise ReadScopeError unless the narrow names a scope this process may read.

    Under ``__ALL__`` the server decides. Otherwise the narrow must name
    allowed channels (by name) or public channels, and must not select DMs
    or use ``with``. Don't subtract hidden scopes with negated terms
    instead: narrows travel in the GET query string, which breaks past a
    few dozen terms.
    """
    if dm_reads_allowed():
        return
    if not all(isinstance(t, dict) for t in narrow):
        raise ReadScopeError("Narrow terms must be dicts with operator and operand.")
    # The server parses a "false" string as False, so only True negates.
    terms = [t for t in narrow if t.get("negated") is not True]
    for term in terms:
        operator, operand = term.get("operator"), term.get("operand")
        if operator in _DM_OPERATORS or (operator == "is" and operand in ("dm", "private")):
            raise ReadScopeError(DM_READS_DENIED)
        # with: moves the narrow to the target message's conversation.
        if operator == "with":
            raise ReadScopeError("The with operator requires BOT_ALLOWED_PRIVATE_STREAMS=__ALL__.")
        # The server resolves an int as a channel ID; the allowlist holds names.
        if operator in _CHANNEL_OPERATORS and not isinstance(operand, str):
            raise ReadScopeError(f"Channel operands must be names, not {type(operand).__name__}.")
    channels = _channel_operands(terms)
    if channels:
        if not all(is_private_stream_allowed(c) for c in channels):
            raise ReadScopeError(PRIVATE_STREAM_DENIED)
        return
    # Not channels:archived, which includes private archived channels.
    if any(t.get("operator") in _CHANNELS_OPERATORS and t.get("operand") in ("public", "web-public")
           for t in terms):
        return
    message = ("Reading without naming a channel requires BOT_ALLOWED_PRIVATE_STREAMS=__ALL__; "
               "narrow to one channel or to public channels.")
    allowlisted = allowlisted_private_streams()
    if allowlisted:
        message += f" Allowlisted private channels: {', '.join(allowlisted)}."
    raise ReadScopeError(message)


class MessagePage(NamedTuple):
    """One page from :func:`fetch_narrow`.

    ``older_than`` is the ``before_message_id`` for the next page, or None
    when nothing older is in range.
    """
    messages: list[dict]
    older_than: Optional[int]


def fetch_narrow(narrow: list[dict], num_messages: int = 20,
                 before_message_id: Optional[int] = None,
                 after: Optional[datetime] = None,
                 before: Optional[datetime] = None) -> MessagePage:
    """Fetch one page of messages matching a narrow, paging backward.

    The page ends just before ``before_message_id`` (exclusive), else just
    before ``before``, else at the newest message; ``after`` is a
    client-side lower bound. Naive bounds are in ``TIMEZONE``; both widen to
    whole seconds, since Zulip timestamps are integers. Unreadable messages,
    and ignored streams unless the narrow names a channel, are dropped after
    the paging cursor is taken.

    Raises ReadScopeError before fetching (see :func:`_check_read_scope`).
    Raises ValueError on API errors, when ``after`` is not earlier than
    ``before``, or when ``before`` is the anchor (no ``before_message_id``)
    on a server older than Zulip 12.0.
    """
    _check_read_scope(narrow)
    if after is not None and after.tzinfo is None:
        after = after.replace(tzinfo=TIMEZONE)
    if before is not None and before.tzinfo is None:
        before = before.replace(tzinfo=TIMEZONE)
    if after is not None and before is not None and after >= before:
        raise ValueError("after must be earlier than before")
    client = get_client()
    limit = min(max(num_messages, 1), 100)
    request: dict = {
        "narrow": narrow,
        "num_before": limit,
        "num_after": 0,
        "apply_markdown": False,
    }
    if before_message_id is not None:
        request["anchor"] = before_message_id
        request["include_anchor"] = False
    elif before is not None:
        if client.feature_level < _DATE_ANCHOR_FEATURE_LEVEL:
            raise ValueError(
                f"before= requires Zulip 12.0+ (feature level {_DATE_ANCHOR_FEATURE_LEVEL}); "
                f"this server is at feature level {client.feature_level}"
            )
        request["anchor"] = "date"
        request["anchor_date"] = before.astimezone(timezone.utc).isoformat()
        # When nothing is on or after the date the server anchors on the
        # newest message, which is in range and must be kept. In the normal
        # case the timestamp filter below drops the anchor.
        request["include_anchor"] = True
    else:
        request["anchor"] = "newest"

    result = client.get_messages(request)
    if result.get("result") != "success":
        raise ValueError(f"Zulip API error: {result.get('msg', 'unknown error')}")
    raw = sorted(result.get("messages", []), key=lambda m: m["id"])

    msgs = raw
    if before is not None:
        cutoff = math.ceil(before.timestamp())
        msgs = [m for m in msgs if m.get("timestamp", 0) < cutoff]
    page = msgs[-limit:]
    trimmed = len(page) < len(msgs)

    reached_after = False
    if after is not None:
        floor = math.floor(after.timestamp())
        in_range = [m for m in page if m.get("timestamp", 0) >= floor]
        reached_after = len(in_range) < len(page)
        page = in_range

    oldest = page or raw
    older_than = None
    if oldest and not reached_after and (trimmed or not result.get("found_oldest", False)):
        older_than = oldest[0]["id"]

    readable = [m for m in page if _is_readable(m)]
    if _ignored_streams and not _channel_operands(narrow):
        readable = [m for m in readable if m.get("type") != "stream"
                    or m.get("display_recipient", "").lower() not in _ignored_streams]
    return MessagePage(readable, older_than)


def send_message(stream: str, topic: str, content: str) -> dict:
    """Send a message. Returns API result dict with 'id' on success."""
    err = _stream_write_error(stream)
    if err:
        return err
    return get_client().send_message({
        "type": "stream",
        "to": stream,
        "subject": topic,
        "content": normalize_zulip_markdown(content),
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
        "content": normalize_zulip_markdown(content),
    })


def add_reaction(message_id: int, emoji_name: str) -> dict:
    """Add emoji reaction to a message. Returns API result dict."""
    return get_client().add_reaction({
        "message_id": message_id,
        "emoji_name": emoji_name,
    })


def remove_reaction(message_id: int, emoji_name: str,
                     reaction_type: Optional[str] = None) -> dict:
    """Remove emoji reaction from a message. Returns API result dict.

    Pass *reaction_type* (``"unicode_emoji"``, ``"realm_emoji"``, or
    ``"zulip_extra_emoji"``) for non-unicode emoji — the server defaults
    to ``"unicode_emoji"`` when omitted, so custom-emoji removals fail
    without it.
    """
    params: dict = {
        "message_id": message_id,
        "emoji_name": emoji_name,
    }
    if reaction_type is not None:
        params["reaction_type"] = reaction_type
    return get_client().remove_reaction(params)


def edit_message(message_id: int, content: str) -> dict:
    """Edit a message's content. Returns API result dict."""
    _, err = _get_stream_message_for_write(message_id)
    if err:
        return err
    return get_client().update_message({
        "message_id": message_id,
        "content": normalize_zulip_markdown(content),
    })


def move_messages(message_id: int, topic: str, stream: Optional[str] = None,
                  propagate_mode: str = "change_one", notify: bool = True) -> dict:
    """Move message(s) to a different topic and/or stream.

    Uses the Zulip update_message API with topic/stream_id params.

    Args:
        message_id: The anchor message to move.
        topic: Destination topic name.
        stream: Destination stream name (optional, only for cross-channel moves).
        propagate_mode: "change_one", "change_later", or "change_all".
        notify: Send notifications to old and new threads (default True).
            Set to False for silent renames like topic resolution.

    Returns:
        API result dict.
    """
    _, err = _get_stream_message_for_write(message_id)
    if err:
        return err

    request: dict = {
        "message_id": message_id,
        "topic": topic,
        "propagate_mode": propagate_mode,
        "send_notification_to_old_thread": notify,
        "send_notification_to_new_thread": notify,
    }
    if stream:
        err = _stream_write_error(stream)
        if err:
            return err
        client = get_client()
        result = client.get_stream_id(stream)
        if result["result"] != "success":
            return result
        request["stream_id"] = result["stream_id"]
    return get_client().update_message(request)


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


def get_emoji_info() -> tuple[int, set[str]]:
    """Return (count, names) of active custom emoji.

    Single API call — use this instead of separate count/has checks.
    Returns ``(0, set())`` on error.
    """
    try:
        result = get_client().get_realm_emoji()
        if result.get("result") != "success":
            return 0, set()
        names = {
            info["name"]
            for info in result.get("emoji", {}).values()
            if not info.get("deactivated", False)
        }
        return len(names), names
    except Exception:
        return 0, set()


def get_emoji_count() -> int:
    """Return count of active custom emoji. Returns 0 on error."""
    count, _ = get_emoji_info()
    return count


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
    target = get_message_by_id(message_id)
    if target is None or target.get("type") != "stream":
        return None
    stream = target.get("display_recipient", "")
    topic = target.get("subject", "")
    if not stream or not topic:
        return None
    return stream, topic


def get_custom_profile_fields() -> list[dict]:
    """Get the org's custom profile field definitions. Cached aggressively."""
    cache_key = ("custom_profile_fields",)
    cached = _get_cache().get(cache_key)
    if cached is not None:
        return cached
    result = get_client().call_endpoint(url="/realm/profile_fields", method="GET")
    if result.get("result") != "success":
        return []
    fields = result.get("custom_fields", [])
    _get_cache().set(cache_key, fields, expire=3600)
    return fields


def get_user_info(email: str) -> Optional[dict]:
    """Get user info by email, including custom profile fields. Returns user dict or None."""
    # Quoted so "/" or "?" in the email can't steer the request to another
    # endpoint once the client joins it onto the base URL.
    result = get_client().call_endpoint(
        url=f"/users/{urllib.parse.quote(email, safe='@+')}",
        method="GET",
        request={"include_custom_profile_fields": True},
    )
    if result.get("result") != "success":
        return None
    return result.get("user")


def get_message_by_id(message_id: int) -> Optional[dict]:
    """Get a message by ID, or None if it doesn't exist or `_is_readable` rejects it.

    apply_markdown=False makes the content field raw markdown (consistent
    with get_topic_messages), not rendered HTML. Unlike an anchored
    get_messages, this endpoint finds public-channel history the account
    never received and never substitutes the nearest message.
    """
    result = get_client().call_endpoint(
        url=f"/messages/{int(message_id)}", method="GET", request={"apply_markdown": False},
    )
    if result.get("result") != "success" or not result.get("message"):
        return None
    msg = result["message"]
    if not _is_readable(msg):
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
    ts_str = datetime.fromtimestamp(timestamp, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else "Unknown"
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


_hash_replacements = {"%": ".", "(": ".28", ")": ".29", ".": ".2E"}


def _encode_hash_component(s: str) -> str:
    """Encode a string for Zulip URL hash fragments (MediaWiki-style dot-encoding)."""
    encoded = urllib.parse.quote(s, safe="")
    return "".join(_hash_replacements.get(c, c) for c in encoded)


def get_message_link(message_id: int) -> str:
    """Return a Zulip markdown link to a message: [#stream > topic](url).

    The URL links to the full conversation context with the specific message
    focused, using the format:
        {base}/#narrow/channel/{stream_id}-{stream_slug}/topic/{topic_encoded}/near/{message_id}

    Falls back to a simpler URL format if stream_id lookup fails.
    """
    client = get_client()
    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]

    msg = get_message_by_id(message_id)
    if not msg:
        url = f"{base}/#narrow/id/{message_id}"
        return f"[message]({url})"
    if msg.get("type") == "private":
        return f"[DM]({base}/#narrow/id/{message_id})"

    stream = msg.get("display_recipient", "")
    topic = msg.get("subject", "")

    stream_id = None
    try:
        result = client.get_stream_id(stream)
        if result.get("result") == "success":
            stream_id = result["stream_id"]
    except Exception:
        pass

    if stream_id:
        topic_encoded = _encode_hash_component(topic)
        stream_slug = _encode_hash_component(stream.replace(" ", "-"))
        url = f"{base}/#narrow/channel/{stream_id}-{stream_slug}/topic/{topic_encoded}/near/{message_id}"
    else:
        url = f"{base}/#narrow/id/{message_id}"

    return f"[#{stream} > {topic}]({url})"


def get_subscribed_streams() -> list[dict]:
    """Get streams this account is subscribed to. Returns list of subscription dicts."""
    result = get_client().get_subscriptions()
    if result["result"] != "success":
        return []
    subs = result.get("subscriptions", [])
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS not in allowed:
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


def _upload_path(path: str) -> str:
    """Validate a ``/user_uploads/`` path and return it.

    Anything else would turn the authenticated session into a generic GET
    proxy that bypasses the private-stream allowlist and bot filtering.
    """
    parts = urllib.parse.urlsplit(path)
    raw = path.lower()
    decoded = urllib.parse.unquote(parts.path)
    # requests sends %2F and %5C as-is and the web server decodes them into
    # separators, so ``..%2F`` would escape /user_uploads/ after this check.
    if (parts.scheme or parts.netloc or parts.query or parts.fragment
            or not parts.path.startswith("/user_uploads/")
            or "%2f" in raw or "%5c" in raw or "\\" in raw or "\\" in decoded
            or any(seg in (".", "..") for seg in decoded.split("/"))):
        raise ValueError(
            f"Only /user_uploads/ paths without a query, dot segments, or encoded "
            f"separators can be fetched: {path!r}"
        )
    return parts.path


def _upload_filename(path: str) -> str:
    """Local filename for an upload path: its basename, never hidden or empty."""
    name = _upload_path(path).rsplit("/", 1)[-1]
    if not name or name.startswith(".") or Path(name).name != name:
        raise ValueError(f"Cannot derive a safe filename from {path!r}")
    return name


def download_file(path: str) -> tuple[bytes, str]:
    """Download a ``/user_uploads/`` file from Zulip. Returns (content_bytes, content_type).

    Access to an upload is by its path token; the stream allowlist isn't checked.
    Raises ValueError for other paths and on any download error.
    """
    path = _upload_path(path)
    client = get_client()
    client.ensure_session()
    if client.session is None:
        raise ValueError("Failed to create Zulip session")

    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]

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
    """Download a Zulip file and save it locally without overwriting.

    Returns (saved_path, size_bytes, content_type). Raises ValueError for
    invalid paths, download errors, or when the target file already exists.
    """
    filename = _upload_filename(path)
    if save_dir:
        save_path = Path(save_dir) / filename
        if save_path.exists():
            raise ValueError(f"File already exists: {save_path}")
    else:
        save_path = Path(tempfile.mkdtemp()) / filename

    content, content_type = download_file(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(save_path, "xb") as f:
            f.write(content)
    except FileExistsError:
        raise ValueError(f"File already exists: {save_path}") from None
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


# ============================================================================
# Event queue — long-polling for real-time message delivery
# ============================================================================

def is_bot_subscribed(stream: str) -> bool:
    """Check if the bot is currently subscribed to a stream."""
    result = get_client().get_subscriptions()
    if result.get("result") != "success":
        return False
    return any(sub["name"].lower() == stream.lower()
               for sub in result.get("subscriptions", []))


def ensure_subscribed(stream: str) -> bool:
    """Subscribe the bot to a stream. No-op if already subscribed.

    For public streams, auto-subscribes. For private streams (or streams
    the bot can't see), returns True only if already subscribed.
    """
    # Fast path: already subscribed
    if is_bot_subscribed(stream):
        return True
    # Private streams can't be joined — must be invited
    if is_stream_private(stream):
        return False
    # Try to subscribe (works for public streams)
    result = get_client().add_subscriptions(
        streams=[{"name": stream}],
    )
    return result.get("result") == "success"


def register_event_queue(stream: str, topic: str) -> tuple[str, int, int]:
    """Register an event queue narrowed to a stream/topic.

    Returns (queue_id, last_event_id, longpoll_timeout_seconds).
    Raises ValueError on failure.
    """
    result = get_client().call_endpoint(
        url="/register",
        method="POST",
        request={
            "event_types": json.dumps(["message", "reaction"]),
            "narrow": json.dumps([
                ["channel", stream],
                ["topic", topic],
            ]),
            "apply_markdown": False,
        },
    )
    if result.get("result") != "success":
        raise ValueError(f"Failed to register event queue: {result.get('msg', '')}")
    timeout = result.get("event_queue_longpoll_timeout_seconds", 90)
    return result["queue_id"], result["last_event_id"], timeout


def get_events(queue_id: str, last_event_id: int,
               longpoll_timeout: int = 90) -> tuple[list[dict], list[dict], int]:
    """Long-poll for events. Blocks until events arrive or server heartbeat.

    Returns (messages, reactions, new_last_event_id). Lists may be empty
    (heartbeat only). Raises ValueError on BAD_EVENT_QUEUE_ID (caller
    should re-register).
    """
    try:
        result = get_client().call_endpoint(
            url="/events",
            method="GET",
            request={
                "queue_id": queue_id,
                "last_event_id": last_event_id,
            },
            timeout=longpoll_timeout + 30,  # HTTP timeout > server timeout
        )
    except requests.exceptions.ReadTimeout:
        return [], [], last_event_id
    if result.get("result") != "success":
        code = result.get("code", "")
        if code == "BAD_EVENT_QUEUE_ID":
            raise ValueError("BAD_EVENT_QUEUE_ID")
        raise ValueError(f"get_events failed: {result.get('msg', '')}")
    events = result.get("events", [])
    messages = []
    reactions = []
    new_last = last_event_id
    for ev in events:
        new_last = max(new_last, ev.get("id", new_last))
        if ev.get("type") == "message":
            # Message events carry flags beside the message, not inside it.
            msg = ev["message"]
            if "flags" in ev:
                msg = {**msg, "flags": ev["flags"]}
            messages.append(msg)
        elif ev.get("type") == "reaction":
            reactions.append(ev)
    return messages, reactions, new_last


def delete_event_queue(queue_id: str) -> None:
    """Delete an event queue. Ignores errors (best-effort cleanup)."""
    try:
        get_client().call_endpoint(
            url="/events",
            method="DELETE",
            request={"queue_id": queue_id},
        )
    except Exception:
        pass
