"""Groq backup for the desk's free model chain (OpenAI-compatible chat API, httpx).

Exposes the same ``interactions.create(**kw)`` shape as ``desk.gemini.GeminiRest`` so the
Gemini backend's model chain and code-run search work unchanged. Used only after every
Gemini model is out of its free daily quota.

Free tier (checked 2026-11-03): about 1,000 requests a day and 8,000 tokens per minute. A
request too large for the per-minute budget (413, or a 429 asking to wait long) fails fast
as a quota error, so the chain moves on; a short Retry-After is waited out once.
"""

import re
import time
from types import SimpleNamespace as NS

import httpx

URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_WAIT_S = 20.0


_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _wait_s(r: httpx.Response) -> float:
    """Seconds Groq asks us to wait: the Retry-After header, else 'try again in 1m2.5s'."""
    if r.headers.get("retry-after"):
        return float(r.headers["retry-after"])
    m = _TRY_AGAIN.search(r.text)
    return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 0.0


class GroqHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail[:300]}")
        self.status_code = status_code


class GroqRest:
    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self._http = httpx.Client(transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})
        self._sleep = sleep
        self.interactions = NS(create=self._create)

    def _create(self, *, model: str, input: str, system_instruction: str = "",
                timeout: float = 180.0, generation_config: dict | None = None,
                tools=None, store=None):
        if tools:
            raise ValueError("Groq backup has no built-in web search")   # caller falls back
        g = generation_config or {}
        body = {"model": model, "max_completion_tokens": min(g.get("max_output_tokens", 4000),
                                                             4000),
                "messages": [{"role": "system", "content": system_instruction},
                             {"role": "user", "content": input}]}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = {"minimal": "low"}.get(g.get("thinking_level"),
                                                              g.get("thinking_level") or "medium")
        for attempt in (0, 1, 2):
            r = self._http.post(URL, json=body, timeout=timeout)
            wait = _wait_s(r)
            if r.status_code == 429 and attempt < 2 and 0 < wait <= MAX_WAIT_S:
                self._sleep(wait + 0.5)          # per-minute budget refills quickly
                continue
            break
        if r.status_code == 413 or (r.status_code == 429 and "tokens per minute" in r.text):
            raise GroqHTTPError(413, r.text)    # too large for the minute: next model, this once
        if r.status_code == 429:
            raise GroqHTTPError(429, r.text)    # out of daily quota: next model
        if r.status_code != 200:
            raise GroqHTTPError(r.status_code, r.text)
        d = r.json()
        u = d.get("usage") or {}
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return NS(status="completed" if text else "incomplete", model=d.get("model") or model,
                  output_text=text,
                  usage=NS(total_input_tokens=u.get("prompt_tokens"),
                           total_output_tokens=u.get("completion_tokens"),
                           total_thought_tokens=0, total_cached_tokens=0,
                           grounding_tool_count=[]))
