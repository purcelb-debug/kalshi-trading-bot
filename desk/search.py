"""Free web search for the desk's News Checker and Skeptic (Tavily, 1,000 free searches a
month), run by code rather than by the model.

The model writes the queries; this module runs them, counts them against a daily cap kept
in a small JSON file (so the monthly free credits last), and formats the results as a
numbered source list. Everything returned from the web is untrusted text: it reaches the
model inside a block that says so, and it can only ever influence a probability, never a
size or a limit (see desk.validate).
"""

import calendar
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

URL = "https://api.tavily.com/search"
ET = ZoneInfo("America/New_York")      # the trading day, as in the executor
SNIPPET_CHARS = 1500


def clean(text: str) -> str:
    """Drop page furniture (menus, "Loading map...", link labels): short lines carry it,
    while forecasts, scores and data releases are full sentences."""
    return " | ".join(ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 25)


class SearchUnavailable(RuntimeError):
    pass


class TavilySearch:
    def __init__(self, api_key: str, *, daily_cap: int, usage_path: str,
                 monthly_cap: int | None = None, max_results: int = 3,
                 transport: httpx.BaseTransport | None = None,
                 clock=lambda: datetime.now(UTC)):
        self._key, self.daily_cap, self.max_results = api_key, daily_cap, max_results
        self.monthly_cap = monthly_cap
        self._usage, self._clock = Path(usage_path), clock
        self._http = httpx.Client(transport=transport, timeout=30)
        self._cache: dict[str, list[dict]] = {}

    def _now(self) -> datetime:
        return self._clock().astimezone(ET)

    def _today(self) -> str:
        return self._now().date().isoformat()

    def _counts(self) -> dict:
        return json.loads(self._usage.read_text(encoding="utf-8")) if self._usage.is_file()             else {}

    def used_today(self) -> int:
        return int(self._counts().get(self._today(), 0))

    def allowance(self) -> int:
        """Searches allowed so far today. The day's share is the month's remainder spread
        over the days left (never more than daily_cap), released evenly through the day so
        the evening's meetings get searches too."""
        now, today = self._now(), self._today()
        cap = self.daily_cap
        if self.monthly_cap is not None:
            before = sum(v for k, v in self._counts().items()
                         if k[:8] == today[:8] and k < today)
            days_left = calendar.monthrange(now.year, now.month)[1] - now.day + 1
            cap = min(cap, max(0, self.monthly_cap - before) // days_left)
        frac = (now.hour * 3600 + now.minute * 60 + now.second + 1) / 86400
        return min(cap, math.ceil(cap * frac))

    def _count(self) -> None:
        d = {k: v for k, v in self._counts().items() if k >= self._today()[:8]}  # this month
        d[self._today()] = d.get(self._today(), 0) + 1
        self._usage.parent.mkdir(parents=True, exist_ok=True)
        self._usage.write_text(json.dumps(d), encoding="utf-8")

    def search(self, query: str) -> list[dict]:
        norm_q = query.strip().lower()
        if norm_q in self._cache:
            return self._cache[norm_q]
        if not self._key:
            raise SearchUnavailable("TAVILY_API_KEY is not set")
        if self.used_today() >= self.daily_cap:
            raise SearchUnavailable(f"daily search cap {self.daily_cap} reached")
        if self.used_today() >= self.allowance():
            raise SearchUnavailable(f"search allowance so far today ({self.allowance()}) "
                                    "used; more are released through the day")
        r = self._http.post(URL, headers={"Authorization": f"Bearer {self._key}"},
                            json={"query": query[:400], "search_depth": "basic",
                                  "max_results": self.max_results, "topic": "general",
                                  "include_published_date": True})
        if r.status_code != 200:
            # 432/433: free plan used up; 429: too fast; 401: bad key. None are retried.
            raise SearchUnavailable(f"Tavily HTTP {r.status_code}: {r.text[:160]}")
        self._count()
        hits = [{"title": x.get("title") or "", "url": x.get("url") or "",
                 "date": x.get("published_date") or "", "content": (x.get("content") or "")}
                for x in r.json().get("results") or []]
        self._cache[norm_q] = hits
        return hits

    def run(self, queries: list[str]) -> tuple[str, int, str]:
        """(numbered source list, searches made, note for the transcript)."""
        lines, n, notes = [], 0, []
        for q in queries:
            try:
                hits = self.search(q)
            except SearchUnavailable as e:
                notes.append(str(e))
                break
            n += 1
            lines.append(f'Query: "{q}"')
            for h in hits:
                k = sum(1 for x in lines if x.startswith("[S"))
                lines.append(f"[S{k + 1}] {h['title']} ({h['date'] or 'date unknown'}) "
                             f"{h['url']}\n    {clean(h['content'])[:SNIPPET_CHARS]}")
        note = f"(web search limited: {'; '.join(notes)})\n" if notes else ""
        return "\n".join(lines) or "No results.", n, note
