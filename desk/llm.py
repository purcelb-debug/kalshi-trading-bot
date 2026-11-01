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
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
