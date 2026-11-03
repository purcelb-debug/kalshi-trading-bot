"""LLM desk: validation, the meeting loop (fake client, no network), and a full desk
cycle on synthetic data through the shared paper record."""

import json
import re
from datetime import timedelta
from types import SimpleNamespace as NS

import pandas as pd
import pytest
from sqlalchemy import select

from desk.config import DeskSettings
from desk.cycle import desk_meetings, register_desk, run_desk_cycle
from desk.meeting import ask, run_meeting
from desk.validate import parse_trades
from executor.gate import evaluate_gate
from predlab.data.collectors.synthetic import DEFAULT_SPECS, generate
from predlab.db.schema import forecasts
from predlab.loop.runner import LoopConfig
from tests.test_phase5 import AS_OF, SERIES, START


def _claude(**kw):
    return DeskSettings(**{"_env_file": None, "provider": "claude", "model": "claude-opus-5-5",
                           "search_provider": "claude", "search_model": "claude-opus-5-5",
                           "skeptic_model": "claude-opus-5-5", **kw})


def _gemini(**kw):
    return DeskSettings(**{"_env_file": None, "provider": "gemini", "search_provider": "gemini",
                           "search_model": "gemini-3.8-flash",
                           "skeptic_model": "gemini-3.8-flash", **kw})


ALLOWED = {"A-1": "EV-A", "A-2": "EV-A", "B-1": "EV-B", "C-1": "EV-C", "D-1": "EV-D"}


def _lead(rows):
    return "Summary.\nTRADES:\n" + json.dumps(rows)


def test_parse_trades_keeps_valid_buys_and_clips():
    ds, notes = parse_trades(_lead([
        {"ticker": "A-1", "action": "buy", "side": "yes", "p_yes": 0.999, "reason": "r"},
        {"ticker": "B-1", "action": "WAIT", "p_yes": 0.4},
    ]), ALLOWED, max_buys=3, p_floor=0.02)
    assert [(d.ticker, d.p_yes) for d in ds] == [("A-1", 0.98)] and notes == []


def test_parse_trades_drops_hostile_or_broken_rows():
    ds, notes = parse_trades(_lead([
        {"ticker": "ZZZ-9", "action": "BUY", "p_yes": 0.9},          # not on the table
        {"ticker": "B-1", "action": "BUY", "p_yes": 5.0},            # impossible p
        {"ticker": "C-1", "action": "BUY"},                          # no p
        {"ticker": "A-1", "action": "BUY", "p_yes": 0.7},
        {"ticker": "A-2", "action": "BUY", "p_yes": 0.7},            # same event
        "garbage",
    ]), ALLOWED, max_buys=3, p_floor=0.02)
    assert [d.ticker for d in ds] == ["A-1"] and len(notes) == 5


def test_parse_trades_caps_buys_and_rejects_missing_list():
    rows = [{"ticker": t, "action": "BUY", "p_yes": 0.6} for t in ("A-1", "B-1", "C-1", "D-1")]
    ds, notes = parse_trades(_lead(rows), ALLOWED, max_buys=2, p_floor=0.02)
    assert len(ds) == 2 and sum("over 2" in n for n in notes) == 2
    assert parse_trades("I think we should buy A-1.", ALLOWED, max_buys=3, p_floor=0.02)[0] == []
    assert parse_trades("TRADES: [oops", ALLOWED, max_buys=3, p_floor=0.02)[0] == []


class FakeClient:
    """Answers by role (matched on the system prompt); records every request."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.answer(kw)
        if isinstance(out, NS):
            return out
        return _resp(out)


def _resp(text, stop="end_turn"):
    return NS(stop_reason=stop, model="claude-opus-5-5",
              content=[NS(type="text", text=text)],
              usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                       server_tool_use=NS(web_search_requests=1)))


def test_ask_resumes_pause_turn_and_handles_refusal():
    seq = iter([_resp("part one ", "pause_turn"), _resp("part two")])
    c = FakeClient(lambda kw: next(seq))
    text, usage = ask(c, _claude(), "NEWS", "sys", "chat", model="m", effort="medium",
                      search=True)
    assert text == "part one part two" and len(usage) == 2
    assert c.calls[1]["messages"][1]["role"] == "assistant"     # resumed, no extra user turn
    assert c.calls[0]["fallbacks"] == "default" and c.calls[0]["tools"][0]["name"] == "web_search"
    c2 = FakeClient(lambda kw: NS(**{**vars(_resp("")), "stop_reason": "refusal"}))
    assert "declined" in ask(c2, _claude(), "RISK", "s", "c", model="m",
                             effort="medium", search=False)[0]


def test_meeting_stops_when_scanner_sees_nothing():
    c = FakeClient(lambda kw: "NOTHING UNUSUAL")
    m = run_meeting({"claude": c}, _claude(), "table")
    assert len(c.calls) == 1 and m.transcript == [("SCANNER", "NOTHING UNUSUAL")]


def test_skeptic_uses_its_own_model():
    c = FakeClient(lambda kw: "ok")
    run_meeting({"claude": c}, _claude(skeptic_model="claude-sonnet-5-5"), "table")
    models = [k["model"] for k in c.calls]
    assert models[4] == "claude-sonnet-5-5" and models.count("claude-sonnet-5-5") == 1


def _buy_everything(kw):
    """A desk that wants every market on the table at p_yes = 0.97."""
    if "DESK LEAD" not in kw["system"]:
        return "notes"
    tickers = re.findall(r"^(\S+) \| ", kw["messages"][0]["content"], re.MULTILINE)[1:]
    return _lead([{"ticker": t, "action": "BUY", "side": "yes", "p_yes": 0.97, "reason": "x"}
                  for t in tickers])


@pytest.fixture
def synth(market_engine):
    generate(market_engine, seed=3, start=START, days=170,
             specs={"weather": DEFAULT_SPECS["weather"]})
    return market_engine


def test_desk_cycle_records_buys_once_and_builds_paper_record(synth, sim_engine, tmp_path):
    cfg = _claude(report_dir=str(tmp_path / "reports"), max_buys=3)
    dep = register_desk(sim_engine, cfg, series=SERIES, horizon_h=6.0)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    client = FakeClient(_buy_everything)
    outs = []
    t = AS_OF
    while t <= AS_OF + pd.Timedelta(days=3):
        outs.append(run_desk_cycle(synth, sim_engine, cfg, loop, clients={"claude": client}, now=t))
        t += timedelta(hours=1)
    held = [o for o in outs if o["desk"]["status"] == "meeting_held"]
    assert held, "no meeting was held"
    with sim_engine.connect() as c:
        fc = c.execute(select(forecasts.c.ticker, forecasts.c.p)
                       .where(forecasts.c.model_version == dep)).all()
        rows = c.execute(select(desk_meetings)).mappings().all()
    shown = [tk for r in rows for tk in r["tickers_json"]]
    assert len(shown) == len(set(shown))                       # each market shown once
    assert fc and all(p == pytest.approx(0.97) for _, p in fc)
    assert len(fc) <= 3 * len(rows)
    assert sum(o["desk"]["recorded"] for o in outs) == len(fc)
    assert (tmp_path / "reports" / "desk-meetings").is_dir()
    gate = outs[-1]["gate"]
    assert gate["paper_days"] >= 3 and gate["paper_settled_trades"] > 0
    assert not evaluate_gate(dep, gate, 1, 1).open             # never certifiable


def test_meeting_cap_blocks_extra_meetings(synth, sim_engine, tmp_path):
    from desk.cycle import init_desk_db
    cfg = _claude(report_dir=str(tmp_path / "r"),
                       max_meetings_per_day=1)
    dep = register_desk(sim_engine, cfg, series=SERIES, horizon_h=6.0)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    client = FakeClient(_buy_everything)
    day = pd.Timestamp(AS_OF).normalize() + pd.Timedelta(days=1)
    init_desk_db(sim_engine)
    with sim_engine.begin() as c:                      # today's one meeting already happened
        c.execute(desk_meetings.insert().values(deployment=dep, ts=day.to_pydatetime(),
                                                tickers_json=[], transcript=""))
    statuses = [run_desk_cycle(synth, sim_engine, cfg, loop, clients={"claude": client},
                               now=day + timedelta(hours=h))["desk"]["status"]
                for h in range(1, 24)]
    assert "meeting_cap_reached" in statuses and "meeting_held" not in statuses
    assert client.calls == []                          # no paid calls past the cap


def test_register_refuses_duplicates(sim_engine):
    cfg = _claude()
    register_desk(sim_engine, cfg, series="X", horizon_h=30)
    with pytest.raises(ValueError):
        register_desk(sim_engine, cfg, series="X", horizon_h=30)


class FakeGemini:
    """Interactions API stand-in; ``fail_search`` mimics a free-tier key."""

    def __init__(self, answer, fail_search=False, status="completed"):
        self.answer, self.fail_search, self.status, self.calls = answer, fail_search, status, []
        self.interactions = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail_search and kw.get("tools"):
            raise RuntimeError("search grounding not available")
        return NS(status=self.status, model=kw["model"], output_text=self.answer(kw),
                  usage=NS(total_input_tokens=10, total_output_tokens=5, total_thought_tokens=3,
                           total_cached_tokens=None,
                           grounding_tool_count=[NS(count=2)] if kw.get("tools") else None))


def test_gemini_meeting_uses_search_on_news_and_skeptic_only():
    c = FakeGemini(lambda kw: "ok")
    m = run_meeting({"gemini": c}, _gemini(skeptic_model="gemini-x"),
                    "table")
    with_search = [i for i, k in enumerate(c.calls) if k.get("tools")]
    assert with_search == [1, 4] and c.calls[4]["model"] == "gemini-x"
    assert all(k["store"] is False and k["system_instruction"] for k in c.calls)
    assert c.calls[4]["generation_config"]["thinking_level"] == "high"
    assert m.usage[1]["web_searches"] == 2 and m.usage[0]["output_tokens"] == 8


def test_gemini_falls_back_without_search_and_handles_no_answer():
    from desk.gemini import ask as gask
    c = FakeGemini(lambda kw: "news", fail_search=True)
    text, usage = gask(c, _gemini(), "NEWS", "s", "c", model="g", effort="medium",
                       search=True)
    assert text.startswith("(web search unavailable") and text.endswith("news")
    assert len(c.calls) == 2 and "tools" not in c.calls[1] and len(usage) == 1
    blocked = FakeGemini(lambda kw: None, status="failed")
    assert "no opinion" in gask(blocked, _gemini(), "RISK", "s", "c",
                                model="g", effort="medium", search=False)[0]


def test_gemini_desk_registers_under_its_own_name(sim_engine):
    dep = register_desk(sim_engine, _gemini(), series="X", horizon_h=30)
    assert dep == "desk@X-h30"


def test_mixed_desk_sends_only_search_roles_to_claude():
    claude, gem = FakeClient(lambda kw: "c"), FakeGemini(lambda kw: "g")
    cfg = DeskSettings(_env_file=None, search_provider="claude", search_model="claude-sonnet-5-5",
                       skeptic_model="claude-sonnet-5-5")
    m = run_meeting({"claude": claude, "gemini": gem}, cfg, "table")
    assert [r for r, _ in m.transcript] == ["SCANNER", "NEWS", "PRICER", "RISK", "SKEPTIC",
                                            "DESK LEAD"]
    assert [msg for _, msg in m.transcript] == ["g", "c", "g", "g", "c", "g"]
    assert all(k.get("tools") for k in claude.calls) and len(claude.calls) == 2
    assert not any(k.get("tools") for k in gem.calls) and len(gem.calls) == 4
    assert {k["model"] for k in gem.calls} == {"gemini-3.8-flash"}
    assert claude.calls[1]["output_config"]["effort"] == "high"          # the Skeptic


def test_meeting_cap_is_shared_across_desk_deployments(sim_engine):
    from desk.cycle import desk_deployments, init_desk_db, meetings_today
    a = register_desk(sim_engine, _claude(), series="AAA", horizon_h=30)
    b = register_desk(sim_engine, _claude(), series="BBB", horizon_h=30)
    assert desk_deployments(sim_engine) == [a, b]
    init_desk_db(sim_engine)
    now = pd.Timestamp("2026-11-01 15:00", tz="UTC")
    with sim_engine.begin() as c:
        for dep in (a, b, b):
            c.execute(desk_meetings.insert().values(deployment=dep, ts=now.to_pydatetime(),
                                                    tickers_json=[], transcript=""))
    assert meetings_today(sim_engine, now) == 3
    assert meetings_today(sim_engine, now + pd.Timedelta(days=1)) == 0


def _scan_rows():
    close = (pd.Timestamp("2026-11-02 12:00", tz="UTC") + pd.Timedelta(hours=10)).isoformat()
    rows = []
    for i in range(5):                               # one busy event with 5 markets
        rows.append({"ticker": f"BIG-{i}", "event_ticker": "BIG", "title": "big", "yes_bid": 0.40,
                     "yes_ask": 0.42, "volume": 9000, "volume_24h": 5000 - i,
                     "open_interest": 100, "close_time": close})
    rows += [
        {"ticker": "WIDE", "event_ticker": "W", "title": "w", "yes_bid": 0.30, "yes_ask": 0.45,
         "volume": 9000, "volume_24h": 9000, "open_interest": 1, "close_time": close},
        {"ticker": "CHEAP", "event_ticker": "C", "title": "c", "yes_bid": 0.01, "yes_ask": 0.03,
         "volume": 9000, "volume_24h": 9000, "open_interest": 1, "close_time": close},
        {"ticker": "THIN", "event_ticker": "T", "title": "t", "yes_bid": 0.50, "yes_ask": 0.52,
         "volume": 5, "volume_24h": 5, "open_interest": 1, "close_time": close},
        {"ticker": "OK", "event_ticker": "O", "title": "ok", "yes_bid": 0.60, "yes_ask": 0.63,
         "volume": 500, "volume_24h": 300, "open_interest": 1, "close_time": close},
        {"ticker": "SEEN", "event_ticker": "S", "title": "s", "yes_bid": 0.50, "yes_ask": 0.51,
         "volume": 9000, "volume_24h": 9999, "open_interest": 1, "close_time": close}]
    return rows


def test_scan_filters_and_ranks_in_code():
    from desk.scan import filter_rank
    t = filter_rank(pd.DataFrame(_scan_rows()), DeskSettings(_env_file=None), exclude={"SEEN"},
                    now=pd.Timestamp("2026-11-02 12:00", tz="UTC"))
    assert t["ticker"].tolist() == ["BIG-0", "BIG-1", "OK"]   # 2 per event, filters, ranked
    assert t["hours_to_close"].round(1).eq(10.0).all()


def test_broad_desk_meets_within_budget_and_records_exits(sim_engine, tmp_path):
    from desk.cycle import meeting_cost, spend_today
    cfg = _claude(report_dir=str(tmp_path / "r"), daily_budget_usd=1.0)
    dep = register_desk(sim_engine, cfg, series="ALL", horizon_h=48)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    now = pd.Timestamp("2026-11-02 12:00", tz="UTC")

    def lead(kw):
        if "DESK LEAD" not in kw["system"]:
            return "notes"
        return _lead([{"ticker": "OK", "action": "BUY", "side": "yes", "p_yes": 0.75,
                       "take_profit": 0.72, "stop": 0.50, "reason": "x"},
                      {"ticker": "ZZZ", "action": "BUY", "p_yes": 0.9}])   # not on the table

    client = FakeClient(lead)
    out = run_desk_cycle(None, sim_engine, cfg, loop, clients={"claude": client}, now=now,
                         scan_fetch=_scan_rows)
    assert out["desk"]["status"] == "meeting_held" and out["desk"]["recorded"] == 1
    with sim_engine.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.explanation)
                         .where(forecasts.c.model_version == dep)).all()
    assert [(r[0], r[1]["take_profit"], r[1]["stop"]) for r in rows] == [("OK", 0.72, 0.5)]
    # usage: 6 Opus calls x (10 in, 5 out) + 2 searches
    assert spend_today(sim_engine, cfg, now) == pytest.approx(
        meeting_cost(out["desk"]["usage"], cfg)) and spend_today(sim_engine, cfg, now) > 0
    tight = cfg.model_copy(update={"daily_budget_usd": 0.0001})
    again = run_desk_cycle(None, sim_engine, tight, loop, clients={"claude": client},
                           now=now + pd.Timedelta(minutes=30), scan_fetch=_scan_rows)
    assert again["desk"]["status"] == "meeting_cap_reached"            # budget spent
    assert next(iter((tmp_path / "r" / "desk-meetings").glob("*ALL*"))).read_text(
        encoding="utf-8").startswith("[DESK MEETING] desk@ALL-h48") 


def test_meeting_falls_back_to_claude_when_gemini_is_down():
    claude = FakeClient(lambda kw: "c")
    down = FakeGemini(lambda kw: "g")
    down.interactions = NS(create=lambda **kw: (_ for _ in ()).throw(ValueError("503")))
    m = run_meeting({"claude": claude, "gemini": down}, DeskSettings(_env_file=None), "table")
    assert len(claude.calls) == 6 and m.transcript[0][1].startswith("(gemini unavailable")


def test_parse_trades_keeps_sane_exit_levels_only():
    ds, notes = parse_trades(_lead([
        {"ticker": "A-1", "action": "BUY", "p_yes": 0.7, "take_profit": 0.8, "stop": 0.5},
        {"ticker": "B-1", "action": "BUY", "p_yes": 0.7, "take_profit": 0.5, "stop": 0.49},
        {"ticker": "C-1", "action": "BUY", "p_yes": 0.7}]), ALLOWED, max_buys=3, p_floor=0.02)
    assert [(d.take_profit, d.stop) for d in ds] == [(0.8, 0.5), (None, None), (None, None)]
    assert len(notes) == 1 and "B-1" in notes[0]


def test_meeting_stops_at_deadline_and_never_parses_a_partial_transcript():
    # start, then per role: deadline check + time-limit read. Passes 600s before PRICER.
    ticks = iter([0, 0, 0, 100, 100, 700, 700, 700, 700])
    c = FakeClient(lambda kw: "TRADES:\n" + json.dumps([{"ticker": "A-1", "action": "BUY",
                                                          "p_yes": 0.9}]))
    m = run_meeting({"claude": c}, _claude(), "table", clock=lambda: next(ticks))
    assert [r for r, _ in m.transcript] == ["SCANNER", "NEWS"] and "deadline" in m.aborted
    assert m.lead_text == "" and len(m.usage) == 2                 # spend kept, no trades


def test_meeting_aborts_when_both_providers_fail():
    def boom(**kw):
        raise ValueError("down")
    claude, gem = FakeClient(lambda kw: "c"), FakeGemini(lambda kw: "g")
    claude.beta = NS(messages=NS(create=boom))
    gem.interactions = NS(create=boom)
    m = run_meeting({"claude": claude, "gemini": gem}, DeskSettings(_env_file=None), "table")
    assert m.transcript == [] and "SCANNER failed" in m.aborted


def test_aborted_meeting_is_recorded_with_its_cost_and_markets(sim_engine, tmp_path):
    cfg = _claude(report_dir=str(tmp_path / "r"), meeting_deadline_s=1.0)
    dep = register_desk(sim_engine, cfg, series="ALL", horizon_h=48)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    now = pd.Timestamp("2026-11-02 12:00", tz="UTC")
    ticks = iter([0.0, 0.0, 5.0] + [5.0] * 20)
    import desk.meeting as dm
    real = dm.time.monotonic
    dm.time.monotonic = lambda: next(ticks)
    try:
        out = run_desk_cycle(None, sim_engine, cfg, loop, clients={"claude": FakeClient(
            lambda kw: "notes")}, now=now, scan_fetch=_scan_rows)
    finally:
        dm.time.monotonic = real
    assert out["desk"]["status"] == "meeting_aborted" and out["desk"]["recorded"] == 0
    with sim_engine.connect() as c:
        row = c.execute(select(desk_meetings.c.tickers_json, desk_meetings.c.usage_json,
                               desk_meetings.c.transcript)).one()
    assert row[0] == ["SEEN", "BIG-0", "BIG-1", "OK"] and len(row[1]) == 1 and "ABORTED" in row[2]
    later = run_desk_cycle(None, sim_engine, cfg, loop, clients={"claude": FakeClient(
        lambda kw: "x")}, now=now, scan_fetch=_scan_rows, allow_meeting=False)
    assert later["desk"]["status"] == "deferred"          # per-run meeting limit respected
    from desk.cycle import _shown
    from desk.scan import filter_rank
    rest = filter_rank(pd.DataFrame(_scan_rows()), cfg, exclude=_shown(sim_engine, dep), now=now)
    assert rest["ticker"].tolist() == ["BIG-2", "BIG-3"]          # aborted markets count as seen


def test_hung_provider_is_cut_off_and_the_role_moves_to_the_other():
    import time as _t

    from desk.meeting import bounded
    with pytest.raises(TimeoutError):
        bounded(lambda: _t.sleep(5), 1.0)
    hang = FakeGemini(lambda kw: "g")
    hang.interactions = NS(create=lambda **kw: _t.sleep(30))      # SDK stuck retrying
    claude = FakeClient(lambda kw: "NOTHING UNUSUAL")
    cfg = DeskSettings(_env_file=None, role_timeout_s=1.0)
    started = _t.monotonic()
    m = run_meeting({"claude": claude, "gemini": hang}, cfg, "table")
    assert _t.monotonic() - started < 10
    assert m.transcript[0][1].startswith("(gemini unavailable: TimeoutError")


def test_gemini_quota_moves_to_the_next_model_then_to_claude():
    from desk import gemini as g

    class Quota(Exception):
        status_code = 429

    g.EXHAUSTED.clear()
    out_of_quota = {"gemini-3.8-flash"}

    def create(**kw):
        if kw["model"] in out_of_quota:
            raise Quota("limit: 20 requests per day")
        return NS(status="completed", model=kw["model"], output_text=f"from {kw['model']}",
                  usage=NS(total_input_tokens=1, total_output_tokens=1, total_thought_tokens=0,
                           total_cached_tokens=0, grounding_tool_count=None))

    gem = NS(interactions=NS(create=create))
    cfg = DeskSettings(_env_file=None)
    text, usage = g.ask(gem, cfg, "SCANNER", "s", "c", model=cfg.model, effort="low",
                        search=False)
    assert text == "from gemini-3.6-flash" and usage[0]["model"] == "gemini-3.6-flash"
    assert g.EXHAUSTED == {"gemini-3.8-flash"}                     # not retried this run
    out_of_quota.update(cfg.gemini_models)
    claude = FakeClient(lambda kw: "NOTHING UNUSUAL")
    m = run_meeting({"claude": claude, "gemini": gem}, cfg, "table")
    assert m.transcript[0][1].startswith("(gemini unavailable: QuotaExhausted")
    g.EXHAUSTED.clear()


def test_gemini_rest_client_parses_answers_and_fails_fast_on_quota():
    import httpx

    from desk import gemini as g
    seen = []

    def handler(req):
        body = json.loads(req.read())
        seen.append((req.headers["x-goog-api-key"], body["model"], "timeout" in body))
        if body["model"] == "gemini-3.8-flash":
            return httpx.Response(429, headers={"retry-after": "45536"},
                                  json={"error": {"message": "limit: 20 requests per day"}})
        return httpx.Response(200, json={
            "status": "completed", "model": body["model"],
            "usage": {"total_input_tokens": 9, "total_output_tokens": 1,
                      "total_thought_tokens": 96, "total_cached_tokens": 0},
            "steps": [{"type": "thought", "signature": "x"},
                      {"type": "model_output", "content": [{"type": "text", "text": "Ok"}]}]})

    g.EXHAUSTED.clear()
    client = g.GeminiRest("k", transport=httpx.MockTransport(handler))
    cfg = DeskSettings(_env_file=None)
    text, usage = g.ask(client, cfg, "SCANNER", "s", "c", model=cfg.model, effort="low",
                        search=False)
    assert text == "Ok" and usage[0]["model"] == "gemini-3.6-flash"
    assert usage[0]["output_tokens"] == 97                       # thinking billed as output
    assert seen == [("k", "gemini-3.8-flash", False), ("k", "gemini-3.6-flash", False)]
    g.EXHAUSTED.clear()


def test_free_desk_searches_by_code_and_never_calls_claude(tmp_path):
    import httpx

    from desk import gemini as g
    from desk.search import TavilySearch
    searched = []

    def tavily(req):
        body = json.loads(req.read())
        searched.append((req.headers["authorization"], body["query"]))
        return httpx.Response(200, json={"results": [
            {"title": "NWS forecast", "url": "https://weather.gov/x", "content": "High 55F",
             "published_date": "2026-11-03"}]})

    def answer(kw):
        if "SEARCH PLAN" in kw["system_instruction"]:
            return 'QUERIES: ["NWS Central Park high Nov 3", "second", "third", "fourth"]'
        if "WEB SEARCH RESULTS" in kw["input"]:
            assert "[S1] NWS forecast (2026-11-03) https://weather.gov/x" in kw["input"]
            return "High 55F per NWS [S1]"
        return "NOTHING UNUSUAL" if "SCANNER" in kw["system_instruction"] else "ok"

    g.EXHAUSTED.clear()
    gem = FakeGemini(answer)
    gem.searcher = TavilySearch("tvly-k", daily_cap=30, usage_path=str(tmp_path / "u.json"),
                                transport=httpx.MockTransport(tavily))
    cfg = DeskSettings(_env_file=None, searches_per_role=3)
    assert cfg.provider == cfg.search_provider == "gemini"
    text, usage = g.ask(gem, cfg, "NEWS", "You are the NEWS CHECKER.", "table",
                        model=cfg.search_model, effort="medium", search=True)
    assert text.startswith("(searched: NWS Central Park high Nov 3; second; third)")
    assert text.endswith("High 55F per NWS [S1]") and len(searched) == 3
    assert searched[0] == ("Bearer tvly-k", "NWS Central Park high Nov 3")
    assert usage[1]["web_searches"] == 3 and usage[1]["search_backend"] == "tavily"
    from desk.cycle import meeting_cost
    assert meeting_cost(usage, cfg) == 0.0                         # free end to end
    m = run_meeting({"gemini": gem}, cfg, "table")                  # no Claude client at all
    assert m.transcript[0] == ("SCANNER", "NOTHING UNUSUAL")


def test_search_cap_and_quota_errors_degrade_to_no_search(tmp_path):
    import httpx

    from desk.search import TavilySearch
    calls = []

    def tavily(req):
        calls.append(1)
        return (httpx.Response(432, json={"detail": {"error": "plan limit"}}) if len(calls) > 2
                else httpx.Response(200, json={"results": []}))

    s = TavilySearch("k", daily_cap=5, usage_path=str(tmp_path / "u.json"),
                     transport=httpx.MockTransport(tavily))
    _, n, note = s.run(["a", "b", "c", "d"])
    assert n == 2 and "Tavily HTTP 432" in note and s.used_today() == 2
    capped = TavilySearch("k", daily_cap=2, usage_path=str(tmp_path / "u.json"),
                          transport=httpx.MockTransport(tavily))
    _, n2, note2 = capped.run(["x"])
    assert n2 == 0 and "daily search cap 2 reached" in note2 and len(calls) == 3
    assert TavilySearch("", daily_cap=5, usage_path=str(tmp_path / "v.json")).run(["q"])[1] == 0


def test_parse_queries_is_strict():
    from desk.gemini import parse_queries
    assert parse_queries('blah\nQUERIES: ["a", " b ", 3, ""]', 5) == ["a", "b"]
    assert parse_queries("QUERIES: [oops", 5) == [] and parse_queries("no plan", 5) == []
