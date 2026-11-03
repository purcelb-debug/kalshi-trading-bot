"""LLM client for the desk, by ``DeskSettings.provider``.

Claude uses ANTHROPIC_API_KEY from the environment or the repository .env (read by
predlab's settings). Gemini uses GEMINI_API_KEY (repository .env or the environment).
Otherwise each SDK's own credential lookup."""

from desk.config import DeskSettings


def make_client(cfg: DeskSettings, provider: str):
    if provider == "gemini":
        from desk.gemini import GeminiRest
        from desk.search import TavilySearch

        client = GeminiRest(cfg.gemini_api_key)
        if cfg.tavily_api_key:          # free search run by code for News and Skeptic
            client.searcher = TavilySearch(cfg.tavily_api_key,
                                           daily_cap=cfg.search_daily_cap,
                                           usage_path=cfg.search_usage_path)
        if cfg.groq_api_key:            # free backup once Gemini's daily quotas run out
            from desk.groq import GroqRest
            client.backup = GroqRest(cfg.groq_api_key)
        return client

    import anthropic

    from predlab.config import get_settings

    key = get_settings().anthropic_api_key
    # Retries for dropped connections and 529s; each attempt is bounded by call_timeout_s.
    return (anthropic.Anthropic(api_key=key, max_retries=2) if key
            else anthropic.Anthropic(max_retries=2))
