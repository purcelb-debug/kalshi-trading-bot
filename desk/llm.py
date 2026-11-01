"""Anthropic client for the desk. Uses ANTHROPIC_API_KEY from the environment or the
repository .env (read by predlab's settings); otherwise the SDK's own credential lookup."""


def make_client():
    import anthropic

    from predlab.config import get_settings

    key = get_settings().anthropic_api_key
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
