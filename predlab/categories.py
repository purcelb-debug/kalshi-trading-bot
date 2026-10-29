"""The five research categories and the mapping from Kalshi's own category labels."""

WEATHER = "weather"
ECONOMICS = "economics"
FINANCIAL = "financial"
SPORTS = "sports"
PUBLIC_EVENTS = "public_events"

CATEGORIES = (WEATHER, ECONOMICS, FINANCIAL, SPORTS, PUBLIC_EVENTS)

_KALSHI_MAP = {
    "climate and weather": WEATHER,
    "economics": ECONOMICS,
    "financials": FINANCIAL,
    "crypto": FINANCIAL,
    "sports": SPORTS,
}


def map_kalshi_category(label: str | None) -> str:
    """Anything not explicitly mapped is a publicly announced, objectively verifiable event."""
    if not label:
        return PUBLIC_EVENTS
    return _KALSHI_MAP.get(label.strip().lower(), PUBLIC_EVENTS)
