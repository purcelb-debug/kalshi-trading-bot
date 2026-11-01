"""One desk meeting: six LLM calls in order, each reading the whole chat so far.

``ask`` below is the Claude backend and ``desk.gemini.ask`` the Gemini one. News Checker and
Skeptic need web search and run on ``search_provider``; the other four roles run on
``provider``. Every Claude call opts into
server-side refusal fallbacks (``fallbacks="default"``), so a safety decline is retried
on Anthropic's recommended model in the same request. Nothing here sizes or places
anything; the transcript is just text for ``desk.validate`` to parse.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from desk.config import DeskSettings

ROLES = (  # (chat label, prompt key, uses web search, is the skeptic)
    ("SCANNER", "scanner", False, False),
    ("NEWS", "news", True, False),
    ("PRICER", "pricer", False, False),
    ("RISK", "risk", False, False),
    ("SKEPTIC", "skeptic", True, True),
    ("DESK LEAD", "lead", False, False),
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_CONTINUATIONS = 5


@dataclass
class Meeting:
    transcript: list[tuple[str, str]] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)

    @property
    def lead_text(self) -> str:
        return self.transcript[-1][1] if self.transcript else ""

    def text(self) -> str:
        return "\n\n".join(f"[{name}]\n{msg}" for name, msg in self.transcript)


def load_prompts(cfg: DeskSettings) -> dict:
    """Reloaded every meeting, so prompt edits apply on the next one."""
    p = json.loads(Path(cfg.prompts_path).read_text(encoding="utf-8"))
    return {k: v.replace("{max_picks}", str(cfg.max_picks)) for k, v in p.items()}


def _usage(resp, role: str) -> dict:
    u = resp.usage
    return {"role": role, "model": resp.model, "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "web_searches": getattr(getattr(u, "server_tool_use", None),
                                    "web_search_requests", 0) or 0}


def ask(client, cfg: DeskSettings, role: str, system: str, chat: str, *, model: str,
        effort: str, search: bool) -> tuple[str, list[dict]]:
    """Claude: one role's message and the usage of every request it took."""
    kw = {"model": model, "max_tokens": 16000,
          "system": system, "betas": [FALLBACK_BETA], "fallbacks": "default",
          "output_config": {"effort": effort}}
    if search:
        kw["tools"] = [{"type": "web_search_20260209", "name": "web_search",
                        "max_uses": cfg.search_max_uses}]
    messages = [{"role": "user", "content": chat}]
    usage, text = [], []
    for _ in range(MAX_CONTINUATIONS + 1):
        resp = client.beta.messages.create(messages=messages, **kw)
        usage.append(_usage(resp, role))
        if resp.stop_reason == "refusal":
            return "(declined to answer; treat as no opinion)", usage
        text += [b.text for b in resp.content if b.type == "text"]
        if resp.stop_reason != "pause_turn":
            break
        # Server-side search loop paused: resend with the partial turn and it resumes.
        messages = [{"role": "user", "content": chat},
                    {"role": "assistant", "content": resp.content}]
    return "".join(text).strip() or "(no answer)", usage


def providers(cfg: DeskSettings) -> set[str]:
    return {cfg.provider, cfg.search_provider}


def run_meeting(clients: dict, cfg: DeskSettings, opening: str,
                prompts: dict | None = None) -> Meeting:
    """Run the six roles over ``opening`` (the code-built market table). ``clients`` maps
    each provider in ``providers(cfg)`` to its SDK client."""
    from desk import gemini

    backends = {"claude": ask, "gemini": gemini.ask}
    prompts = prompts or load_prompts(cfg)
    m, chat = Meeting(), opening
    for name, key, search, skeptic in ROLES:
        prov = cfg.search_provider if search else cfg.provider
        model = cfg.skeptic_model if skeptic else cfg.search_model if search else cfg.model
        msg, usage = backends[prov](clients[prov], cfg, name, prompts[key], chat, model=model,
                                    effort=cfg.skeptic_effort if skeptic else cfg.effort,
                                    search=search)
        m.transcript.append((name, msg))
        m.usage += usage
        chat += f"\n\n[{name}]\n{msg}"
        if key == "scanner" and "NOTHING UNUSUAL" in msg.upper():
            break        # nothing to check: skip the paid search calls
    return m
