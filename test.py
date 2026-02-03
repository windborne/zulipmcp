from datetime import datetime
from zoneinfo import ZoneInfo
from zulip_core import get_messages_formatted

if __name__ == "__main__":
    # Last Friday 9am PT, looking back one full week
    friday_9am = datetime(2026, 1, 30, 9, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    print(get_messages_formatted(hours_back=168, sender="John Dean", as_of=friday_9am))
