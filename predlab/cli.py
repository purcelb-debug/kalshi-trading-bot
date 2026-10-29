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
):
    """Download public Kalshi series/events/markets (optionally trades, candles, books)."""
    from predlab.data.collectors.kalshi import KalshiCollector, make_client

    eng = _market_engine()

    async def run():
        async with make_client(get_settings()) as client:
            col = KalshiCollector(eng, client)
            for s in series:
                st = await col.collect_series_bundle(
                    s, status=status or None, with_trades=trades, with_candles=candles,
                    with_orderbooks=books)
                typer.echo(f"{s}: {st.rows} (issues recorded: {st.issues}, "
                           f"requests: {client.requests_made})")

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


if __name__ == "__main__":
    app()
