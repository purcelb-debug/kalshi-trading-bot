"""One desk meeting: six LLM calls in order, each reading the whole chat so far.

``ask`` below is the Claude backend and ``desk.gemini.ask`` the Gemini one. News Checker and
Skeptic need web search and run on ``search_provider``; the other four roles run on
``provider``. Every Claude call opts into
server-side refusal fallbacks (``fallbacks="default"``), so a safety decline is retried
on Anthropic's recommended model in the same request. Nothing here sizes or places
anything; the transcript is just text for ``desk.validate`` to parse.
"""

import json
import time
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
    aborted: str = ""            # non-empty: the meeting stopped early, with the reason

    @property
    def lead_text(self) -> str:
        """The Desk Lead's message, or "" if the meeting never reached the Lead (so a
        partial transcript can never be parsed as trades)."""
        if self.aborted or not self.transcript or self.transcript[-1][0] != "DESK LEAD":
            return ""
        return self.transcript[-1][1]

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
        resp = client.beta.messages.create(messages=messages, timeout=cfg.call_timeout_s, **kw)
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
                prompts: dict | None = None, clock=None) -> Meeting:
    """Run the six roles over ``opening`` (the code-built market table). ``clients`` maps
    each provider in ``providers(cfg)`` to its SDK client."""
    from desk import gemini

    backends = {"claude": ask, "gemini": gemini.ask}
    prompts = prompts or load_prompts(cfg)
    clock = clock or time.monotonic
    m, chat, started = Meeting(), opening, clock()
    for name, key, search, skeptic in ROLES:
        if clock() - started > cfg.meeting_deadline_s:
            m.aborted = f"meeting deadline {cfg.meeting_deadline_s:g}s passed before {name}"
            break
        prov = cfg.search_provider if search else cfg.provider
        model = cfg.skeptic_model if skeptic else cfg.search_model if search else cfg.model
        effort = cfg.skeptic_effort if skeptic else cfg.effort
        try:
            msg, usage = backends[prov](clients[prov], cfg, name, prompts[key], chat,
                                        model=model, effort=effort, search=search)
        except Exception as e:  # noqa: BLE001 - one provider down: the other answers
            other = "claude" if prov == "gemini" else "gemini"
            alt = cfg.search_model if other == cfg.search_provider else cfg.model
            try:
                if other not in clients:
                    raise
                msg, usage = backends[other](clients[other], cfg, name, prompts[key], chat,
                                             model=alt, effort=effort, search=search)
            except Exception as e2:  # noqa: BLE001 - both down: stop, keep what was spent
                m.aborted = f"{name} failed: {type(e).__name__}; fallback {type(e2).__name__}"
                break
            msg = f"({prov} unavailable: {type(e).__name__}; answered on {other})\n" + msg
        m.transcript.append((name, msg))
        m.usage += usage
        chat += f"\n\n[{name}]\n{msg}"
        if key == "scanner" and "NOTHING UNUSUAL" in msg.upper():
            break        # nothing to check: skip the paid search calls
    return m
