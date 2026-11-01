"""Send text to one Telegram chat. Failures are reported, never raised: a notifier
outage must not break the loop that called it."""

import httpx

from notify.config import NotifySettings

LIMIT = 4000   # Telegram rejects messages over 4096 characters


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) > limit:
            out.append(cur)
            cur = ""
        cur += line
    if cur:
        out.append(cur)
    return out


def send(cfg: NotifySettings, text: str, client: httpx.Client | None = None) -> tuple[bool, str]:
    """(delivered, error). Does nothing and returns (False, 'disabled') without a token."""
    if not cfg.enabled:
        return False, "disabled"
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        for part in chunks(text):
            r = client.post(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
                            data={"chat_id": cfg.telegram_chat_id, "text": part})
            if r.status_code != 200:
                return False, f"telegram {r.status_code}: {r.text[:200]}"
        return True, ""
    except httpx.HTTPError as e:
        return False, f"telegram error: {type(e).__name__}"   # never echo the token URL
    finally:
        if own:
            client.close()
