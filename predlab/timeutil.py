"""UTC timestamp normalization. Every timestamp stored by predlab is timezone-aware UTC."""

from datetime import UTC, date, datetime

__all__ = ["UTC", "to_utc", "utcnow"]



def to_utc(value) -> datetime | None:
    """Normalize ISO strings, epoch seconds, dates, or datetimes to aware UTC.

    Naive datetimes are assumed to already be UTC (documented assumption).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return to_utc(datetime.fromisoformat(s))
    raise TypeError(f"cannot convert {type(value).__name__} to UTC datetime")


def utcnow() -> datetime:
    return datetime.now(UTC)
