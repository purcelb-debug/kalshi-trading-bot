"""Phase 5: certification, the forward paper loop, degradation and gate evidence."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from predlab.backtest.engine import StrategyConfig
from predlab.data.collectors.synthetic import DEFAULT_SPECS, generate
from predlab.db.schema import forecasts, holdout_evaluations, model_registry, research_log
from predlab.evaluation.splits import HoldoutAlreadyUsed
from predlab.features.pointintime import build_live_dataset
from predlab.forecasts import verify_chain
from predlab.loop.promotion import certify, deployment_version
from predlab.loop.runner import (
    DeploymentError,
    LoopConfig,
    check_degradation,
    load_deployment,
    replay,
    run_cycle,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
SERIES = "SYN-WEATHER"
H = 6.0
AS_OF = pd.Timestamp(START + timedelta(days=150))


@pytest.fixture
def synth(market_engine):
    generate(market_engine, seed=3, start=START, days=170,
             specs={"weather": DEFAULT_SPECS["weather"]})
    return market_engine


@pytest.fixture
def certified(synth, sim_engine):
    c = certify(synth, sim_engine, model="logistic_l1", series=SERIES, horizon_h=H,
                as_of=AS_OF, n_boot=200)
    return c


def test_certify_records_one_look_and_gate_evidence(synth, sim_engine, certified):
    c = certified
    assert c.deployment.endswith(f"@{SERIES}-h6")
    assert c.status in ("promoted", "rejected")
    for k in ("holdout_brier_diff_vs_market_ci_high", "holdout_net_pnl_ci_low", "shrink_kappa"):
        assert k in c.evidence
    assert c.evidence["paper_days"] == 0 and c.evidence["certification_rank"] == 1
    with sim_engine.connect() as conn:
        reg = conn.execute(select(model_registry)).mappings().one()
        keys = {r.model_version for r in conn.execute(select(holdout_evaluations))}
    assert reg["status"] == c.status and reg["params_json"]["series"] == SERIES
    assert keys == {c.deployment, reg["params_json"]["model_version"]}
    with pytest.raises(HoldoutAlreadyUsed):   # one look per deployment, ever
        certify(synth, sim_engine, model="logistic_l1", series=SERIES, horizon_h=H,
                as_of=AS_OF + pd.Timedelta(days=5), n_boot=200)


def test_second_model_faces_bonferroni(synth, sim_engine, certified):
    c2 = certify(synth, sim_engine, model="frequency_bayes", series=SERIES, horizon_h=H,
                 as_of=AS_OF, n_boot=200)
    assert c2.evidence["certification_rank"] == 2
    assert c2.evidence["ci_level"] == pytest.approx(1 - 0.05 / 2)


def test_live_dataset_window_and_no_lookahead(synth):
    now = AS_OF + pd.Timedelta(hours=14)        # synthetic markets close at 20:00 UTC
    ds = build_live_dataset(synth, series=SERIES, as_of=now, feature_names=["x1", "x2", "x3"],
                            min_to_close=timedelta(hours=4), max_to_close=timedelta(hours=6))
    assert len(ds) > 0 and np.isnan(ds.y).all()
    assert list(ds.X.columns) == ["x1", "x2", "x3"]
    assert (pd.to_datetime(ds.meta["input_max_ts"], utc=True) <= now).all()
    late = build_live_dataset(synth, series=SERIES, as_of=now + pd.Timedelta(hours=3),
                              feature_names=["x1"], min_to_close=timedelta(hours=4),
                              max_to_close=timedelta(hours=6))
    assert len(late) == 0                       # past the decision window


def test_replay_forecasts_once_and_builds_paper_record(synth, sim_engine, certified):
    cfg = LoopConfig(certified.deployment, report_dir=str(synth.url.database) + "-reports")
    out = replay(synth, sim_engine, cfg, start=AS_OF, end=AS_OF + pd.Timedelta(days=6),
                 step=timedelta(hours=1))
    assert all(o["status"] == "ok" for o in out)
    with sim_engine.connect() as c:
        rows = c.execute(select(forecasts.c.ticker, forecasts.c.created_ts,
                                forecasts.c.input_max_ts)
                         .where(forecasts.c.model_version == cfg.deployment)).all()
    tickers = [r.ticker for r in rows]
    assert tickers and len(tickers) == len(set(tickers))          # once per market
    assert all(pd.Timestamp(r.input_max_ts) <= pd.Timestamp(r.created_ts) for r in rows)
    hours = {pd.Timestamp(r.created_ts).hour for r in rows}
    assert hours <= {14, 15, 16}                                  # inside [h - 2h, h]
    assert verify_chain(sim_engine) == []
    gate = out[-1]["gate"]
    assert gate["paper_days"] == 7 and gate["paper_resolved"] > 0
    assert gate["paper_forecasts"] == len(tickers)
    stored = load_deployment(sim_engine, cfg.deployment)["validation_json"]["gate"]
    assert stored["paper_days"] == 7
    # Re-running the same hour adds no forecasts and rebuilds the same paper record.
    again = run_cycle(synth, sim_engine, cfg, now=AS_OF + pd.Timedelta(days=6))
    assert again["new_forecasts"] == 0
    assert again["paper"]["net_pnl"] == out[-1]["paper"]["net_pnl"]


def test_degradation_alerts():
    rng = np.random.default_rng(0)
    n = 60
    y = rng.integers(0, 2, n).astype(float)
    scored = pd.DataFrame({"created_ts": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
                           "p": 1 - y * 0.9 - 0.05,          # confidently wrong
                           "market_ref_p": np.where(y == 1, 0.8, 0.2), "y": y,
                           "cluster": [f"E{i}" for i in range(n)]})
    alerts, _ = check_degradation(scored, None, LoopConfig("d"), StrategyConfig())
    assert [a["rule"] for a in alerts] == ["worse_than_market"]
    few, _ = check_degradation(scored.head(10), None, LoopConfig("d"), StrategyConfig())
    assert few == []                                     # too few to judge


def test_alerts_logged_once_per_day(synth, sim_engine, certified, monkeypatch):
    import predlab.loop.runner as r
    monkeypatch.setattr(r, "check_degradation",
                        lambda *a, **k: ([{"rule": "paper_drawdown", "detail": "x"}], {"resolved": 0}))
    cfg = LoopConfig(certified.deployment, report_dir=str(synth.url.database) + "-reports")
    replay(synth, sim_engine, cfg, start=AS_OF, end=AS_OF + pd.Timedelta(hours=30),
           step=timedelta(hours=6))
    with sim_engine.connect() as c:
        msgs = [m for (m,) in c.execute(select(research_log.c.message)
                                        .where(research_log.c.component == "degradation"))]
    assert len(msgs) == 2                                # two UTC dates
    assert load_deployment(sim_engine, cfg.deployment)["validation_json"]["gate"][
        "paper_degradation_alerts"] == 2


def test_uncertified_or_changed_model_refused(synth, sim_engine, certified, monkeypatch):
    with pytest.raises(DeploymentError):
        run_cycle(synth, sim_engine, LoopConfig("nope@X-h1"), now=AS_OF)
    with sim_engine.begin() as c:
        row = c.execute(select(model_registry)).mappings().one()
        c.execute(model_registry.update().values(params_json={**row["params_json"],
                                                              "model_version": "other-123"}))
    with pytest.raises(DeploymentError, match="certify the new version"):
        run_cycle(synth, sim_engine, LoopConfig(certified.deployment), now=AS_OF)


def test_deployment_version_format():
    assert deployment_version("logistic_l2-abc", "KXHIGHNY", 30.0) == "logistic_l2-abc@KXHIGHNY-h30"


def test_replay_fills_settle_and_persist(synth, sim_engine, certified):
    """Force trading (trust the model fully, no edge threshold) to exercise the fill path."""
    from predlab.db.schema import sim_fills, sim_positions
    from predlab.loop.runner import paper_run_id
    with sim_engine.begin() as c:
        row = c.execute(select(model_registry)).mappings().one()
        v = dict(row["validation_json"])
        v["strategy"] = {**v["strategy"], "shrink": 1.0, "min_edge": 0.0}
        c.execute(model_registry.update().values(validation_json=v))
    cfg = LoopConfig(certified.deployment, report_dir=str(synth.url.database) + "-reports")
    out = replay(synth, sim_engine, cfg, start=AS_OF, end=AS_OF + pd.Timedelta(days=8),
                 step=timedelta(hours=1))
    run = paper_run_id(cfg.deployment)
    with sim_engine.connect() as c:
        fills = c.execute(select(sim_fills).where(sim_fills.c.run_id == run)).mappings().all()
        pos = c.execute(select(sim_positions).where(sim_positions.c.run_id == run)).mappings().all()
        fc = dict(c.execute(select(forecasts.c.ticker, forecasts.c.created_ts)
                            .where(forecasts.c.model_version == cfg.deployment)).all())
    assert fills and out[-1]["paper"]["n_fills"] == len(fills)
    assert all(pd.Timestamp(f["ts"]) == pd.Timestamp(fc[f["ticker"]]) for f in fills)
    assert any(p["settled"] for p in pos)


def test_candidate_trades_on_paper_then_certify_upgrades(synth, sim_engine):
    from predlab.loop.promotion import register_candidate
    cand = register_candidate(synth, sim_engine, model="logistic_l1", series=SERIES,
                              horizon_h=H, as_of=AS_OF)
    assert cand.status == "candidate"
    with sim_engine.connect() as c:
        assert c.execute(select(holdout_evaluations)).first() is None      # holdout untouched
    cfg = LoopConfig(cand.deployment, report_dir=str(synth.url.database) + "-reports")
    replay(synth, sim_engine, cfg, start=AS_OF, end=AS_OF + pd.Timedelta(days=2),
           step=timedelta(hours=6))
    cert = certify(synth, sim_engine, model="logistic_l1", series=SERIES, horizon_h=H,
                   as_of=AS_OF, n_boot=200)
    assert cert.deployment == cand.deployment
    assert cert.evidence["certification_rank"] == 1        # candidates are not holdout looks
    out = run_cycle(synth, sim_engine, cfg, now=AS_OF + pd.Timedelta(days=3))
    assert out["registry_status"] == cert.status
    assert out["gate"]["paper_days"] == 4                  # history kept across the upgrade
    assert out["gate"]["holdout_net_pnl_ci_low"] == cert.evidence["holdout_net_pnl_ci_low"]
    with pytest.raises(HoldoutAlreadyUsed):
        certify(synth, sim_engine, model="logistic_l1", series=SERIES, horizon_h=H,
                as_of=AS_OF, n_boot=200)
