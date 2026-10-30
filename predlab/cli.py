"""predlab command line. Every command reads public data or runs a local simulation only."""

import asyncio
import sys
from datetime import date

import typer
from sqlalchemy import func, select

from predlab import BANNER
from predlab.config import get_settings
from predlab.db.repo import init_market_db, init_sim_db, make_engine
from predlab.db.schema import MARKET_MD, data_quality_issues, source_health
from predlab.logs import configure_logging

app = typer.Typer(add_completion=False, help=f"predlab research platform. {BANNER}.")


@app.callback()
def _main():
    settings = get_settings()
    configure_logging(settings.log_level)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    typer.secho(BANNER, fg=typer.colors.YELLOW, bold=True, err=True)


def _market_engine():
    s = get_settings()
    s.ensure_sqlite_dirs()
    eng = make_engine(s.market_db_url)
    init_market_db(eng)
    return eng


@app.command("init-db")
def init_db():
    """Create both databases (historical market data and simulation data)."""
    s = get_settings()
    s.ensure_sqlite_dirs()
    init_market_db(make_engine(s.market_db_url))
    init_sim_db(make_engine(s.sim_db_url))
    typer.echo(f"market db: {s.market_db_url}\nsim db:    {s.sim_db_url}")


@app.command("ingest-kalshi")
def ingest_kalshi(
    series: list[str] = typer.Option(..., "--series", help="Kalshi series ticker, e.g. KXHIGHNY"),
    status: str = typer.Option("settled", help="unopened|open|closed|settled, or '' for all"),
    trades: bool = False, candles: bool = False, books: bool = False,
    since: str = typer.Option("", help="only fetch trades/candles/books for markets closing "
                                       "on or after this date (YYYY-MM-DD)"),
    max_markets: int = typer.Option(20000, help="stop each tier's market download after this "
                                                "many markets (0 = no cap)"),
):
    """Download public Kalshi series/events/markets (optionally trades, candles, books)."""
    from predlab.data.collectors.kalshi import KalshiCollector, make_client
    from predlab.timeutil import to_utc

    eng = _market_engine()

    async def run():
        async with make_client(get_settings()) as client:
            col = KalshiCollector(eng, client)
            for s in series:
                st = await col.collect_series_bundle(
                    s, status=status or None, with_trades=trades, with_candles=candles,
                    with_orderbooks=books, since=to_utc(since) if since else None,
                    max_markets=max_markets or None)
                typer.echo(f"{s}: {st.rows} (issues recorded: {st.issues}, "
                           f"requests: {client.requests_made})")
                for w in st.warnings:
                    typer.echo(f"  warning: {w}")

    asyncio.run(run())


@app.command("ingest-weather")
def ingest_weather(start: str = typer.Option(...), end: str = typer.Option(...),
                   forecasts: bool = typer.Option(True, help="also snapshot current NWS forecasts")):
    """NCEI observed daily highs/lows (and a point-in-time NWS forecast snapshot)."""
    from predlab.data.collectors.weather import WeatherCollector

    eng = _market_engine()

    async def run():
        col = WeatherCollector(eng, get_settings())
        try:
            st = await col.collect_ncei(date.fromisoformat(start), date.fromisoformat(end))
            typer.echo(f"ncei: {st.rows}")
            if forecasts:
                st = await col.snapshot_nws_forecasts()
                typer.echo(f"nws: {st.rows}")
        finally:
            await col.aclose()

    asyncio.run(run())


@app.command("ingest-fred")
def ingest_fred(series: list[str] = typer.Option(None, "--series")):
    """FRED/ALFRED series with full vintage history (needs PREDLAB_FRED_API_KEY)."""
    from predlab.data.collectors.econ_fred import DEFAULT_SERIES, FredCollector

    eng = _market_engine()

    async def run():
        col = FredCollector(eng, get_settings())
        try:
            st = await col.collect(tuple(series) if series else DEFAULT_SERIES)
            typer.echo("fred: skipped (no API key)" if st is None else f"fred: {st.rows}")
        finally:
            await col.aclose()

    asyncio.run(run())


@app.command("load-sports")
def load_sports(path: str):
    """Load a historical sports results CSV (see predlab/data/collectors/sports_csv.py)."""
    from predlab.data.collectors.sports_csv import load_sports_csv

    st = asyncio.run(load_sports_csv(_market_engine(), path))
    typer.echo(f"sports csv: {st.rows}")


@app.command("synth")
def synth(seed: int = 7, days: int = 365):
    """Generate a deterministic synthetic dataset for all five categories."""
    from predlab.data.collectors.synthetic import generate_tracked

    res = asyncio.run(generate_tracked(_market_engine(), seed=seed, days=days))
    typer.echo(f"synthetic markets: {res.markets}\n{res.rows}")


@app.command("health")
def health():
    """Show data-source health, data-quality issue counts, and table sizes."""
    eng = _market_engine()
    with eng.connect() as conn:
        typer.echo("source health:")
        for r in conn.execute(select(source_health)):
            ok = "OK " if r.consecutive_failures == 0 else "ERR"
            typer.echo(f"  [{ok}] {r.source:<12} last_success={r.last_success} rows={r.rows_last_run}"
                       f" fails={r.consecutive_failures} {r.last_error or ''}")
        typer.echo("data-quality issues by check:")
        q = (select(data_quality_issues.c.source, data_quality_issues.c.check_name, func.count())
             .group_by(data_quality_issues.c.source, data_quality_issues.c.check_name))
        for src, chk, n in conn.execute(q):
            typer.echo(f"  {src:<12} {chk:<34} {n}")
        typer.echo("table rows:")
        for t in MARKET_MD.sorted_tables:
            typer.echo(f"  {t.name:<26} {conn.execute(select(func.count()).select_from(t)).scalar()}")


def _dataset(source, category, series, horizon_hours, no_market_features, since=""):
    from datetime import timedelta

    from predlab.features.pointintime import build_dataset
    from predlab.timeutil import to_utc

    return build_dataset(_market_engine(), source=source or None, category=category or None,
                         series=series or None, horizon=timedelta(hours=horizon_hours),
                         use_market_features=not no_market_features,
                         since=to_utc(since) if since else None)


def _factories(models: str):
    from predlab.models.registry import ALL_MODELS as MODEL_FACTORIES

    names = list(MODEL_FACTORIES) if models == "all" else [m.strip() for m in models.split(",")]
    unknown = set(names) - set(MODEL_FACTORIES)
    if unknown:
        raise typer.BadParameter(f"unknown models: {sorted(unknown)}; "
                                 f"choose from {sorted(MODEL_FACTORIES)}")
    if "market_implied" not in names:
        names.insert(0, "market_implied")  # always keep the reference to beat
    return {n: MODEL_FACTORIES[n] for n in names}


@app.command("evaluate")
def evaluate(source: str = "", category: str = "", series: str = "",
             models: str = "all", folds: int = 5, holdout_frac: float = 0.2,
             horizon_hours: float = 6.0, embargo_hours: float = 0.0, min_train: int = 50,
             no_market_features: bool = False, n_boot: int = 500,
             since: str = typer.Option("", help="only markets closing on/after YYYY-MM-DD")):
    """Walk-forward evaluation (the holdout is carved off and NOT touched)."""
    import json
    from datetime import timedelta
    from pathlib import Path

    from predlab.evaluation.walkforward import run_walk_forward
    from predlab.timeutil import utcnow

    ds = _dataset(source, category, series, horizon_hours, no_market_features, since)
    res = run_walk_forward(ds, _factories(models), n_folds=folds, holdout_frac=holdout_frac,
                           min_train=min_train, embargo=timedelta(hours=embargo_hours),
                           n_boot=n_boot)
    typer.echo(f"rows={len(ds)} features={ds.feature_names}")
    typer.echo(f"folds={len(res.folds)} validated={res.predictions['ticker'].nunique()} "
               f"holdout={len(res.holdout_rows)} (hash {res.holdout_hash[:12]}, untouched)")
    typer.echo(res.table().round(4).to_string())
    typer.echo("vs_market_brier_diff < 0 means better than the market; if the CI spans 0, "
               "the difference is not distinguishable.")
    if res.decided_share > 0.5:
        typer.secho(f"WARNING: {res.decided_share:.0%} of markets were priced <=3c or >=97c at "
                    "decision time, so most outcomes were already effectively known. Either the "
                    "horizon is too short (try a longer --horizon-hours, e.g. 30 for daily "
                    "weather) or most strikes are far from the money (normal for strike ladders).",
                    fg=typer.colors.RED)
    if res.market_coverage < 0.95:
        typer.secho(f"WARNING: only {res.market_coverage:.0%} of validated markets have a "
                    "market price at decision time. For the rest, 'market_implied' is the "
                    "training base rate, so vs_market is NOT a comparison with the market. "
                    "Ingest candles (--candles) for a real market reference.",
                    fg=typer.colors.RED)
    if res.errors:
        typer.secho(f"model errors: {res.errors}", fg=typer.colors.RED)
    out = Path("reports") / f"evaluate-{utcnow():%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"banner": BANNER, "filters": {"source": source,
                               "category": category, "series": series},
                               "versions": res.versions, "holdout_hash": res.holdout_hash,
                               "market_coverage": res.market_coverage,
                               "decided_share": res.decided_share,
                               "summary": res.summary, "errors": res.errors},
                              indent=2, default=str), encoding="utf-8")
    typer.echo(f"report: {out}")


@app.command("holdout")
def holdout(model: str = typer.Option(...), source: str = "", category: str = "",
            series: str = "", holdout_frac: float = 0.2, horizon_hours: float = 6.0,
            embargo_hours: float = 0.0, no_market_features: bool = False):
    """Score ONE model on the untouched holdout. Allowed once per model version."""
    from datetime import timedelta

    from predlab.evaluation.splits import HoldoutGuard, holdout_split
    from predlab.evaluation.walkforward import EvalResult, evaluate_holdout

    s = get_settings()
    sim = make_engine(s.sim_db_url)
    init_sim_db(sim)
    ds = _dataset(source, category, series, horizon_hours, no_market_features)
    dev, hold, h = holdout_split(ds, holdout_frac)
    res = EvalResult(None, {}, [], h, hold, dev)
    make = _factories(model)[model]
    m = evaluate_holdout(ds, res, model, make, HoldoutGuard(sim),
                         embargo=timedelta(hours=embargo_hours))
    typer.echo(f"{m['model_version']} on holdout {h[:12]} (n={m['n']}, "
               f"trained on {m['train_rows']} rows)")
    for k in ("brier", "log_loss", "ece"):
        v = m[k]
        typer.echo(f"  {k:<9} {v['estimate']:.4f}  [{v['low']:.4f}, {v['high']:.4f}]")


@app.command("backtest")
def backtest(model: str = typer.Option(...), source: str = "", category: str = "",
             series: str = "", folds: int = 5, holdout_frac: float = 0.2,
             horizon_hours: float = 6.0, embargo_hours: float = 0.0, min_train: int = 50,
             no_market_features: bool = False, initial_cash: float = 1000.0,
             min_edge: float = 0.03, kelly_fraction: float = 0.25,
             max_market_frac: float = 0.02, slippage: float = 0.01,
             sensitivity: bool = True,
             shrink: str = typer.Option("learned", help="'learned' (per-fold kappa from "
                                        "earlier folds) or a fixed number in [0, 1]")):
    """Paper-trade a model's out-of-sample walk-forward forecasts (holdout untouched)."""
    import json
    from datetime import timedelta
    from pathlib import Path

    from predlab.backtest.engine import (
        StrategyConfig,
        fee_schedules,
        run_backtest,
        save_run,
        sensitivity_grid,
        signals_from_walk_forward,
    )
    from predlab.backtest.execution import ExecConfig, MarketData
    from predlab.backtest.fees import FeeSchedule
    from predlab.evaluation.walkforward import run_walk_forward

    s = get_settings()
    eng = _market_engine()
    sim = make_engine(s.sim_db_url)
    init_sim_db(sim)
    ds = _dataset(source, category, series, horizon_hours, no_market_features)
    wf = run_walk_forward(ds, _factories(model), n_folds=folds, holdout_frac=holdout_frac,
                          min_train=min_train, embargo=timedelta(hours=embargo_hours), n_boot=200)
    sig, res = signals_from_walk_forward(wf, model, ds, learned_shrink=shrink == "learned")
    md = MarketData(eng, sig["ticker"])
    strat = StrategyConfig(initial_cash=initial_cash, min_edge=min_edge,
                           kelly_fraction=kelly_fraction, max_market_frac=max_market_frac,
                           shrink=1.0 if shrink == "learned" else float(shrink))
    if "kappas" in sig.attrs:
        typer.echo(f"learned kappa per fold (trust in model vs market): {sig.attrs['kappas']}")

    def run(fee_stress, slip):
        return run_backtest(sig, res, md, strategy=strat, execution=ExecConfig(slippage=slip),
                            fee_for=fee_schedules(eng, fee_stress),
                            default_fees=FeeSchedule(stress=fee_stress))

    out = run(1.0, slippage)
    run_id = save_run(sim, out, model_version=wf.versions.get(model), holdout_hash=wf.holdout_hash)
    sm = out.summary
    typer.secho(f"\n{BANNER}  run {run_id}  model {wf.versions.get(model)}", bold=True)
    for k in ("n_signals", "n_orders", "n_fills", "contracts", "uncertain_fill_share",
              "gross_pnl_settled", "fees", "net_pnl", "net_pnl_ci95", "return", "max_drawdown",
              "peak_exposure", "hit_rate", "brier_all_signals", "brier_traded",
              "open_positions_at_end", "rejections"):
        v = sm[k]
        typer.echo(f"  {k:<24} {round(v, 4) if isinstance(v, float) else v}")
    if wf.market_coverage < 0.95:
        typer.secho(f"WARNING: market price coverage {wf.market_coverage:.0%}; markets without "
                    "quotes cannot be traded (rejected as no_liquidity_data).",
                    fg=typer.colors.RED)
    grid = sensitivity_grid(run) if sensitivity else None
    if grid is not None:
        typer.echo("\nsensitivity (net P&L by fee multiplier x slippage per contract):")
        typer.echo(grid.pivot(index="fee_stress", columns="slippage", values="net_pnl")
                   .round(2).to_string())
    out_path = Path("reports") / f"backtest-{run_id}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"banner": BANNER, "run_id": run_id, "summary": sm,
                                    "config": out.config, "market_coverage": wf.market_coverage,
                                    "sensitivity": None if grid is None else grid.to_dict("records")},
                                   indent=2, default=str), encoding="utf-8")
    typer.echo(f"report: {out_path}")


@app.command("compare-categories")
def compare_categories_cmd(source: str = "", series: str = "",
                           models: str = "frequency_bayes,logistic_l2,lightgbm,market_platt",
                           folds: int = 5, horizon_hours: float = 6.0, min_rows: int = 100,
                           n_boot: int = 300,
                           since: str = typer.Option("", help="only markets closing on/after "
                                                              "YYYY-MM-DD (all categories)")):
    """Rank categories by forecasting quality and reliability (not a betting ranking)."""
    import json
    from datetime import timedelta
    from pathlib import Path

    import pandas as pd

    from predlab.evaluation.category_rank import DISCLAIMER, compare_categories
    from predlab.timeutil import to_utc, utcnow

    table, skipped = compare_categories(
        _market_engine(), _factories(models), source=source or None, series=series or None,
        horizon=timedelta(hours=horizon_hours), n_folds=folds, min_rows=min_rows, n_boot=n_boot,
        since=to_utc(since) if since else None)
    typer.secho(DISCLAIMER, fg=typer.colors.YELLOW)
    if table.empty:
        typer.echo(f"no category had enough data: {skipped}")
        raise typer.Exit(1)
    cols = ["rank", "category", "n", "best_model", "brier", "log_loss", "ece", "fold_brier_sd",
            "vs_market_brier_diff", "oos_win_share", "independent_info", "market_coverage",
            "est_cost_per_contract", "freshness_days"]
    with pd.option_context("display.width", 250):
        typer.echo(table[cols].round(4).to_string(index=False))
    for cat, why in skipped.items():
        typer.echo(f"skipped {cat}: {why}")
    out = Path("reports") / f"categories-{utcnow():%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"banner": BANNER, "disclaimer": DISCLAIMER,
                               "table": table.to_dict("records"), "skipped": skipped},
                              indent=2, default=str), encoding="utf-8")
    typer.echo(f"report: {out}")


@app.command("stress")
def stress_cmd(model: str = typer.Option(...), source: str = "", category: str = "",
               series: str = "", folds: int = 5, horizon_hours: float = 6.0,
               no_market_features: bool = False, initial_cash: float = 1000.0,
               min_edge: float = 0.03, probe: bool = True,
               shrink: str = typer.Option("learned", help="'learned' or a fixed number")):
    """Stress-test a model's out-of-sample paper trading, plus overfitting probes."""
    import json
    from pathlib import Path

    import pandas as pd

    from predlab.backtest.engine import StrategyConfig, fee_schedules, signals_from_walk_forward
    from predlab.backtest.execution import MarketData
    from predlab.backtest.stress import run_stress_suite
    from predlab.evaluation.overfit import multiple_testing_check, shuffled_label_probe
    from predlab.evaluation.walkforward import run_walk_forward
    from predlab.timeutil import utcnow

    eng = _market_engine()
    ds = _dataset(source, category, series, horizon_hours, no_market_features)
    facs = _factories(model)
    wf = run_walk_forward(ds, facs, n_folds=folds, n_boot=200)
    sig, res = signals_from_walk_forward(wf, model, ds, learned_shrink=shrink == "learned")
    if "kappas" in sig.attrs:
        typer.echo(f"learned kappa per fold: {sig.attrs['kappas']}")
    table = run_stress_suite(sig, res, MarketData(eng, sig["ticker"]),
                             strategy=StrategyConfig(
                                 initial_cash=initial_cash, min_edge=min_edge,
                                 shrink=1.0 if shrink == "learned" else float(shrink)),
                             fee_for=fee_schedules(eng),
                             fee_for_stress=lambda m: fee_schedules(eng, m))
    typer.secho(f"{BANNER}  stress suite for {wf.versions.get(model)}", bold=True)
    with pd.option_context("display.width", 200):
        typer.echo(table.to_string(index=False))
    extra = {}
    if probe:
        extra["shuffled_label_probe"] = shuffled_label_probe(ds, facs, n_folds=min(folds, 3))
        extra["multiple_testing"] = multiple_testing_check(wf)
        typer.echo("")
        typer.echo("shuffled-label probe (gain vs constant on permuted labels; ~0 expected):")
        for m, v in extra["shuffled_label_probe"].items():
            flag = "  <-- SUSPICIOUS: possible leakage" if v["suspicious"] else ""
            typer.echo(f"  {m:<20} {v['mean_gain_vs_constant']:+.5f}{flag}")
        typer.echo("multiple-testing (Bonferroni) edge vs market:")
        for m, v in extra["multiple_testing"].items():
            typer.echo(f"  {m:<20} diff {v['diff']:+.4f} CI [{v['low']:+.4f}, {v['high']:+.4f}] "
                       f"survives={v['survives']}")
    out = Path("reports") / f"stress-{utcnow():%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"banner": BANNER, "model_version": wf.versions.get(model),
                               "stress": table.to_dict("records"), **extra},
                              indent=2, default=str), encoding="utf-8")
    typer.echo(f"report: {out}")


if __name__ == "__main__":
    app()
