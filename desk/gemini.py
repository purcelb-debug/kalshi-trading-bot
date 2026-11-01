"""Gemini backend for one desk role (Interactions API, ``google-genai`` SDK).

Same contract as ``desk.meeting.ask``: the role's text plus one usage row per request.
News Checker and Skeptic get Google Search grounding. Free-tier keys can't use it, so a
refused search request is asked again without the tool and the answer says so.
Requests are sent with ``store=False``: nothing is kept server-side between roles.
"""

from desk.config import DeskSettings


def _usage(resp, role: str, model: str) -> dict:
    u = resp.usage
    n = lambda v: v or 0
    return {"role": role, "model": getattr(resp, "model", None) or model,
            "input_tokens": n(getattr(u, "total_input_tokens", 0)),
            # thinking tokens are billed as output
            "output_tokens": n(getattr(u, "total_output_tokens", 0))
                             + n(getattr(u, "total_thought_tokens", 0)),
            "cache_read_input_tokens": n(getattr(u, "total_cached_tokens", 0)),
            "web_searches": sum(n(g.count) for g in (getattr(u, "grounding_tool_count", None)
                                                     or []))}


def ask(client, cfg: DeskSettings, role: str, system: str, chat: str, *, model: str,
        effort: str, search: bool) -> tuple[str, list[dict]]:
    kw = {"model": model, "input": chat, "system_instruction": system, "store": False,
          "generation_config": {"thinking_level": effort, "max_output_tokens": 16000}}
    note = ""
    if search:
        try:
            resp = client.interactions.create(tools=[{"type": "google_search"}], **kw)
        except Exception as e:  # noqa: BLE001 - e.g. search grounding not on this tier
            note = f"(web search unavailable: {type(e).__name__}; answered without it)\n"
            resp = client.interactions.create(**kw)
    else:
        resp = client.interactions.create(**kw)
    usage = [_usage(resp, role, model)]
    text = (resp.output_text or "").strip() if resp.status == "completed" else ""
    if not text:
        return f"(no answer, status {resp.status}; treat as no opinion)", usage
    return note + text, usage
