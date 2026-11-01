"""Weekly scorecard: trade statistics and a read of a replayed paper record."""

from datetime import timedelta

import pandas as pd
import pytest
from sqlalchemy import select

from predlab.db.schema import model_registry
from predlab.loop.runner import LoopConfig, replay
from predlab.loop.scorecard import format_scorecard, scorecard, trade_stats
from tests.test_phase5 import AS_OF, certified, synth  # noqa: F401 - fixtures


def _trades(nets):
    ts = pd.date_range("2026-10-01", periods=len(nets), freq="D", tz="UTC")
    return pd.DataFrame({"net": nets, "resolve_ts": ts})


def test_trade_stats_known_values():
    s = trade_stats(_trades([2.0, -1.0, -1.0, -1.0, 3.0, -1.0]))
    assert s["trades"] == 6 and s["wins"] == 2 and s["losses"] == 4
    assert s["win_rate"] == pytest.approx(2 / 6)
    assert s["avg_win"] == pytest.approx(2.5) and s["avg_loss"] == pytest.approx(-1.0)
    # expectancy = win_rate * avg_win + loss_rate * avg_loss
    assert s["expectancy"] == pytest.approx(2 / 6 * 2.5 + 4 / 6 * -1.0)
    assert s["worst_losing_streak"] == 3
    assert s["net_pnl"] == pytest.approx(1.0)


def test_trade_stats_orders_by_resolution_time():
    t = _trades([-1.0, 5.0, -1.0]).iloc[[2, 0, 1]]   # shuffled rows
    assert trade_stats(t)["worst_losing_streak"] == 1


def test_trade_stats_empty():
    s = trade_stats(_trades([]))
    assert s["trades"] == 0 and s["expectancy"] is None and s["worst_losing_streak"] == 0


def test_scorecard_reads_replayed_paper_record(synth, sim_engine, certified):  # noqa: F811
    with sim_engine.begin() as c:
        row = c.execute(select(model_registry)).mappings().one()
        v = dict(row["validation_json"])
        v["strategy"] = {**v["strategy"], "shrink": 1.0, "min_edge": 0.0}
        c.execute(model_registry.update().values(validation_json=v))
    cfg = LoopConfig(certified.deployment, report_dir=str(synth.url.database) + "-reports")
    end = AS_OF + pd.Timedelta(days=8)
    out = replay(synth, sim_engine, cfg, start=AS_OF, end=end, step=timedelta(hours=1))
    card = scorecard(synth, sim_engine, cfg.deployment, days=None, now=end)
    assert card["all_time"]["trades"] == out[-1]["gate"]["paper_settled_trades"] > 0
    assert card["window"] == card["all_time"]
    assert card["paper"]["n_fills"] == out[-1]["paper"]["n_fills"]
    assert certified.deployment in format_scorecard(card)
    week = scorecard(synth, sim_engine, cfg.deployment, days=1, now=end)
    assert week["window"]["trades"] <= card["all_time"]["trades"]
