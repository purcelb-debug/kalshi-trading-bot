"""Live underlying values for the broad scan, fetched by code so the desk can price a
threshold market (gold above X, BTC above Y, NYC high above Z) without spending web
searches.

Free, keyless, read-only sources: Coinbase spot (crypto), Yahoo Finance charts (futures,
indices, FX, Treasury yields) and the National Weather Service forecast (temperature and
rain markets). Each value is a reference, not the market's settlement source, which can
differ slightly; the table says so. Any failure leaves that market's cell blank: the scan
never depends on these.
"""

import re
import time
from datetime import UTC, date, datetime, timedelta

import httpx

UA = {"User-Agent": "predlab-desk (personal research)"}
DEADLINE_S = 40.0           # all lookups for one scan together

CRYPTO = re.compile(r"^KX(BTC|ETH|SOL|XRP|DOGE)")
#: Series prefix -> (Yahoo symbol, label). Longest prefixes are matched first.
YAHOO = {
    "KXINX": ("^GSPC", "S&P 500 index"),
    "KXNASDAQ100": ("^NDX", "Nasdaq-100 index"),
    "KXDJI": ("^DJI", "Dow Jones index"),
    "KXWTI": ("CL=F", "WTI crude front-month futures"),
    "KXBRENT": ("BZ=F", "Brent crude front-month futures"),
    "KXGOLD": ("GC=F", "gold front-month futures"),
    "KXSILVER": ("SI=F", "silver front-month futures"),
    "KXCOPPER": ("HG=F", "copper front-month futures"),
    "KXNATGAS": ("NG=F", "natural gas front-month futures"),
    "KXUST10": ("^TNX", "10-year Treasury yield, %"),
    "KXEURUSD": ("EURUSD=X", "EUR/USD"),
    "KXGBPUSD": ("GBPUSD=X", "GBP/USD"),
    "KXAUDUSD": ("AUDUSD=X", "AUD/USD"),
    "KXUSDJPY": ("JPY=X", "USD/JPY"),
}
#: Kalshi weather city codes -> NWS forecast point (city or its main airport; approximate,
#: each market's rules name the exact station).
CITIES = {
    "NY": (40.7789, -73.9692), "NYC": (40.7789, -73.9692), "CHI": (41.7868, -87.7522),
    "MIA": (25.7906, -80.3164), "AUS": (30.3208, -97.7604), "DEN": (39.8467, -104.6562),
    "LAX": (33.9382, -118.3866), "PHIL": (39.8733, -75.2268), "ATL": (33.6297, -84.4422),
    "BOS": (42.3606, -71.0097), "DAL": (32.8998, -97.0403), "DC": (38.8483, -77.0342),
    "EWR": (40.6895, -74.1745), "HOU": (29.6454, -95.2789), "LV": (36.0719, -115.1634),
    "MIN": (44.8831, -93.2289), "NOLA": (29.9934, -90.2580), "OKC": (35.3889, -97.6006),
    "PHX": (33.4278, -112.0037), "SAN": (32.7336, -117.1831), "SATX": (29.5443, -98.4839),
    "SDF": (38.1811, -85.7392), "SEA": (47.4447, -122.3136), "SFO": (37.6197, -122.3656),
    "TTN": (40.2767, -74.8135), "ABQ": (35.0419, -106.6156), "CMH": (39.9950, -82.8783),
    "DTW": (42.2314, -83.3308), "IND": (39.7173, -86.2944), "LEX": (38.0365, -84.6059),
    "MKE": (42.9550, -87.9044), "PIT": (40.4846, -80.2144), "PVD": (41.7240, -71.4283),
    "SGF": (37.2457, -93.3886), "STL": (38.7525, -90.3734), "TPA": (27.9619, -82.5404),
    "WPB": (26.6832, -80.0956),
}
_WEATHER = re.compile(r"^KX(HIGH|LOW)T?([A-Z]+)$")
_DAY = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                         "SEP", "OCT", "NOV", "DEC"], start=1)}


def market_day(ticker: str) -> date | None:
    """The date in a ticker like KXHIGHNY-26NOV05-T70 (year, month, day)."""
    parts = ticker.split("-")
    m = _DAY.match(parts[1]) if len(parts) > 1 else None
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def weather_city(ticker: str) -> tuple[str, str] | None:
    """(kind, city code) for temperature and rain markets, else None."""
    parts = ticker.split("-")
    if parts[0] == "KXRAIN" and len(parts) > 2 and parts[-1] in CITIES:
        return "rain", parts[-1]
    m = _WEATHER.match(parts[0])
    if not m:
        return None
    code = m.group(2)
    if code not in CITIES and code.startswith("T") and code[1:] in CITIES:
        code = code[1:]
    return (m.group(1).lower(), code) if code in CITIES else None


def classify(ticker: str):
    series = ticker.split("-")[0]
    if m := CRYPTO.match(series):
        return ("crypto", m.group(1))
    for prefix in sorted(YAHOO, key=len, reverse=True):
        if series.startswith(prefix):
            return ("yahoo", prefix)
    if w := weather_city(ticker):
        return ("weather", w)
    return None


def _fmt(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 10 else f"{x:,.2f}"


class Spot:
    """One scan's lookups, each source fetched at most once."""

    def __init__(self, get=None, clock=time.monotonic):
        self._get = get or httpx.Client(timeout=8, headers=UA, follow_redirects=True).get
        self._clock, self._memo = clock, {}
        self._stop = clock() + DEADLINE_S

    def _once(self, key, fn):
        if key not in self._memo:
            if self._clock() > self._stop:
                return None
            try:
                self._memo[key] = fn()
            except Exception:  # noqa: BLE001 - optional data: a blank cell, never a failure
                self._memo[key] = None
        return self._memo[key]

    def _json(self, url, params=None):
        r = self._get(url, params=params)
        r.raise_for_status()
        return r.json()

    def crypto(self, coin: str) -> str | None:
        d = self._once(("cb", coin), lambda: self._json(
            f"https://api.coinbase.com/v2/prices/{coin}-USD/spot"))
        return f"{coin} spot {_fmt(float(d['data']['amount']))} USD (Coinbase)" if d else None

    def yahoo(self, prefix: str) -> str | None:
        sym, label = YAHOO[prefix]
        d = self._once(("yf", sym), lambda: self._json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            {"range": "1d", "interval": "5m"})["chart"]["result"][0]["meta"])
        if not d or d.get("regularMarketPrice") is None:
            return None
        at = datetime.fromtimestamp(d["regularMarketTime"], UTC).strftime("%m-%d %H:%M UTC") \
            if d.get("regularMarketTime") else "time unknown"
        return f"{label} {_fmt(float(d['regularMarketPrice']))} (Yahoo {sym}, {at})"

    def _periods(self, code: str):
        def fetch():
            lat, lon = CITIES[code]
            url = self._json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
                             )["properties"]["forecast"]
            return self._json(url)["properties"]["periods"]
        return self._once(("nws", code), fetch)

    def weather(self, kind: str, code: str, day: date | None) -> str | None:
        periods = self._periods(code)
        if not periods or day is None:
            return None
        on = lambda p, d: datetime.fromisoformat(p["startTime"]).date() == d
        if kind == "high":
            p = next((p for p in periods if p["isDaytime"] and on(p, day)), None)
            what = f"high {p['temperature']}°{p['temperatureUnit']}" if p else None
        elif kind == "low":
            p = next((p for p in periods if not p["isDaytime"]
                      and on(p, day - timedelta(days=1))), None)
            what = f"overnight low {p['temperature']}°{p['temperatureUnit']}" if p else None
        else:
            pops = [f"{p['name']} {(p.get('probabilityOfPrecipitation') or {}).get('value') or 0}%"
                    for p in periods if on(p, day)]
            what = "chance of rain " + ", ".join(pops) if pops else None
        return f"NWS forecast {code} {day:%b %d}: {what}" if what else None

    def value(self, ticker: str) -> str | None:
        c = classify(ticker)
        if c is None:
            return None
        if c[0] == "crypto":
            return self.crypto(c[1])
        if c[0] == "yahoo":
            return self.yahoo(c[1])
        kind, code = c[1]
        return self.weather(kind, code, market_day(ticker))


def annotate(tickers: list[str], spot: Spot | None = None) -> dict[str, str]:
    """ticker -> live underlying text, for the markets a source covers."""
    s = spot or Spot()
    out = {}
    for t in tickers:
        v = s.value(t)
        if v:
            out[t] = v
    return out
