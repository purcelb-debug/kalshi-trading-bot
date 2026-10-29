"""Static guard: predlab must contain no path to real-money execution.

Run after every phase. A failure here is a release blocker, not a style issue.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "predlab"
SOURCES = sorted(PKG.rglob("*.py"))
HTTP_MODULE = PKG / "data" / "http.py"

FORBIDDEN = {
    "trading endpoint": re.compile(r"""["'][^"']*/portfolio[^"']*["']|["']/orders?["']""", re.IGNORECASE),
    "exchange SDK": re.compile(r"^\s*(import|from)\s+(kalshi|kalshi_python|ccxt|alpaca|"
                               r"py_clob_client|polymarket)\b", re.IGNORECASE | re.MULTILINE),
    "private-key loading": re.compile(r"load_pem_private_key|PRIVATE KEY-----|rsa\.sign|"
                                      r"KALSHI[_A-Z]*KEY", re.IGNORECASE),
    "live/real-money switch": re.compile(r"\b(LIVE_TRADING|REAL_MONEY|ENABLE_LIVE|"
                                         r"EXECUTE_REAL|CONFIRM_REAL)\w*", re.IGNORECASE),
    "write verb on http client": re.compile(r"\b(httpx|_client|client|self\._client)\."
                                            r"(post|put|patch|delete|request|stream|send)\s*\("),
}


def test_sources_found():
    assert len(SOURCES) > 5


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_forbidden_patterns(path):
    text = path.read_text(encoding="utf-8")
    for label, pat in FORBIDDEN.items():
        m = pat.search(text)
        assert m is None, f"{label} found in {path.name}: {m.group(0)!r}"


def test_only_http_module_constructs_clients():
    pat = re.compile(r"httpx\.(Async)?Client\s*\(|requests\.|urllib\.request|aiohttp")
    offenders = [p.name for p in SOURCES if p != HTTP_MODULE and pat.search(p.read_text("utf-8"))]
    assert offenders == []


def test_http_module_only_issues_get():
    text = HTTP_MODULE.read_text(encoding="utf-8")
    calls = re.findall(r"self\._client\.(\w+)\s*\(", text)
    assert set(calls) <= {"get", "aclose"}, calls


def test_env_template_has_no_exchange_credentials():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    keys = re.findall(r"^([A-Z_]+)=", env, re.MULTILINE)
    assert not [k for k in keys if re.search(r"KALSHI.*(KEY|SECRET)|TRADING|LIVE", k)], keys
