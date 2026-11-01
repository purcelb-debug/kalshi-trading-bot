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
    text, usage = ask(c, DeskSettings(_env_file=None), "NEWS", "sys", "chat", search=True,
                      skeptic=False)
    assert text == "part one part two" and len(usage) == 2
    assert c.calls[1]["messages"][1]["role"] == "assistant"     # resumed, no extra user turn
    assert c.calls[0]["fallbacks"] == "default" and c.calls[0]["tools"][0]["name"] == "web_search"
    c2 = FakeClient(lambda kw: NS(**{**vars(_resp("")), "stop_reason": "refusal"}))
    assert "declined" in ask(c2, DeskSettings(_env_file=None), "RISK", "s", "c", search=False,
                             skeptic=False)[0]


def test_meeting_stops_when_scanner_sees_nothing():
    c = FakeClient(lambda kw: "NOTHING UNUSUAL")
    m = run_meeting(c, DeskSettings(_env_file=None), "table")
    assert len(c.calls) == 1 and m.transcript == [("SCANNER", "NOTHING UNUSUAL")]


def test_skeptic_uses_its_own_model():
    c = FakeClient(lambda kw: "ok")
    run_meeting(c, DeskSettings(_env_file=None, skeptic_model="claude-sonnet-5-5"), "table")
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
    cfg = DeskSettings(_env_file=None, report_dir=str(tmp_path / "reports"), max_buys=3)
    dep = register_desk(sim_engine, cfg, series=SERIES, horizon_h=6.0)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    client = FakeClient(_buy_everything)
    outs = []
    t = AS_OF
    while t <= AS_OF + pd.Timedelta(days=3):
        outs.append(run_desk_cycle(synth, sim_engine, cfg, loop, client=client, now=t))
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
    cfg = DeskSettings(_env_file=None, report_dir=str(tmp_path / "r"), max_meetings_per_day=1)
    dep = register_desk(sim_engine, cfg, series=SERIES, horizon_h=6.0)
    loop = LoopConfig(dep, report_dir=cfg.report_dir)
    client = FakeClient(_buy_everything)
    day = pd.Timestamp(AS_OF).normalize() + pd.Timedelta(days=1)
    init_desk_db(sim_engine)
    with sim_engine.begin() as c:                      # today's one meeting already happened
        c.execute(desk_meetings.insert().values(deployment=dep, ts=day.to_pydatetime(),
                                                tickers_json=[], transcript=""))
    statuses = [run_desk_cycle(synth, sim_engine, cfg, loop, client=client,
                               now=day + timedelta(hours=h))["desk"]["status"]
                for h in range(1, 24)]
    assert "meeting_cap_reached" in statuses and "meeting_held" not in statuses
    assert client.calls == []                          # no paid calls past the cap


def test_register_refuses_duplicates(sim_engine):
    cfg = DeskSettings(_env_file=None)
    register_desk(sim_engine, cfg, series="X", horizon_h=30)
    with pytest.raises(ValueError):
        register_desk(sim_engine, cfg, series="X", horizon_h=30)
