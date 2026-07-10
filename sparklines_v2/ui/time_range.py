"""Datetime parsing and time-range resolution for CLI/UI inputs."""

from __future__ import annotations

import datetime as dt


DEFAULT_WINDOW_HOURS = 8.0


def parse_datetime_text(value: str) -> dt.datetime:
    """Parse an ISO-like datetime string into a naive local datetime."""
    text = value.strip()
    if not text:
        raise ValueError("Datetime cannot be empty.")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed.replace(microsecond=0)


def format_datetime_text(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat(sep=" ")


def resolve_time_range(
    *,
    start_text: str | None,
    end_text: str | None,
    hours: float = DEFAULT_WINDOW_HOURS,
    now: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Resolve CLI or UI inputs into an absolute time range."""
    window = dt.timedelta(hours=float(hours))
    current = (now or dt.datetime.now()).replace(microsecond=0)

    start = parse_datetime_text(start_text) if start_text else None
    end = parse_datetime_text(end_text) if end_text else None

    if start is None and end is None:
        end = current
        start = end - window
    elif start is None:
        start = end - window
    elif end is None:
        end = start + window

    if start >= end:
        raise ValueError("Start time must be earlier than end time.")
    return start, end
