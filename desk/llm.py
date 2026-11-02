"""LLM client for the desk, by ``DeskSettings.provider``.

Claude uses ANTHROPIC_API_KEY from the environment or the repository .env (read by
predlab's settings). Gemini uses GEMINI_API_KEY (repository .env or the environment).
Otherwise each SDK's own credential lookup."""

from desk.config import DeskSettings


def make_client(cfg: DeskSettings, provider: str):
    if provider == "gemini":
        from google import genai

        return genai.Client(api_key=cfg.gemini_api_key) if cfg.gemini_api_key else genai.Client()

    import anthropic

    from predlab.config import get_settings

    key = get_settings().anthropic_api_key
    # More retries than the SDK default: dropped connections and 529s are usually brief.
    return (anthropic.Anthropic(api_key=key, max_retries=4) if key
            else anthropic.Anthropic(max_retries=4))
