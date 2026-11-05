"""desk command line: ``register`` once, then ``cycle`` every hour (scripts/run_loop.py)."""

import sys

import typer

from desk.config import DeskSettings

app = typer.Typer(add_completion=False, help="Six-role LLM desk. PAPER TRADING ONLY.")


def _engines():
    from predlab.config import get_settings
    from predlab.db.repo import init_market_db, init_sim_db, make_engine

    s = get_settings()
    s.ensure_sqlite_dirs()
    market, sim = make_engine(s.market_db_url), make_engine(s.sim_db_url)
    init_market_db(market)
    init_sim_db(sim)
    return market, sim


@app.callback()
def _main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    typer.secho("PAPER TRADING - LLM DESK", fg=typer.colors.YELLOW, bold=True, err=True)


@app.command()
def register(series: str = typer.Option(...), horizon: float = 30.0):
    """Admit the desk for a series to forward paper trading (gate stays closed)."""
    from desk.cycle import init_desk_db, register_desk

    _, sim = _engines()
    init_desk_db(sim)
    typer.echo(register_desk(sim, DeskSettings(), series=series, horizon_h=horizon))


@app.command()
def cycle(deployment: str = typer.Option("desk@KXHIGHNY-h30"),
          all_: bool = typer.Option(False, "--all", help="every registered desk deployment"),
          dry: bool = typer.Option(False, "--dry", help="show the table; no LLM calls"),
          review: str = typer.Option("", help="broad desk: comma-separated tickers to "
                                     "look at again; a BUY it now passes on is withdrawn")):
    """Hold a meeting if markets newly reached their decision point, then refresh the
    desk's paper record. The predlab loop step collects data first."""
    import json

    import pandas as pd

    from desk.cycle import desk_deployments, market_table, opening_text, run_desk_cycle
    from predlab.loop.runner import DeploymentError, LoopConfig, load_deployment
    from predlab.timeutil import utcnow

    cfg, (market, sim) = DeskSettings(), _engines()
    if all_:
        # One failing market must not stop the others; the exit code still reports it.
        failed, held = [], 0
        for dep in desk_deployments(sim):
            # The broad desk meets again on the next unseen markets while this run's
            # meeting allowance lasts; a fixed-series desk has one decision point per run.
            again = True
            while again:
                again = False
                try:
                    out = run_desk_cycle(market, sim, cfg,
                                         LoopConfig(dep, report_dir=cfg.report_dir),
                                         allow_meeting=held < cfg.max_meetings_per_cycle)
                    met = out["desk"]["status"] in ("meeting_held", "meeting_aborted")
                    held += met
                    again = met and dep.startswith("desk@ALL") and \
                        held < cfg.max_meetings_per_cycle
                    typer.echo(json.dumps({"deployment": dep, "desk": out["desk"]},
                                          default=str))
                except Exception as e:  # noqa: BLE001 - reported, then the next market runs
                    failed.append(dep)
                    typer.echo(f"{dep} failed: {type(e).__name__}: {e}", err=True)
        raise typer.Exit(1 if failed else 0)
    loop = LoopConfig(deployment, report_dir=cfg.report_dir)
    if dry:   # works before registration: series and horizon come from the id
        series, h = deployment.split("@", 1)[1].rsplit("-h", 1)
        now = utcnow()
        if series == "ALL":
            from desk.cycle import _shown, init_desk_db
            from desk.scan import scan_table, scan_text
            init_desk_db(sim)
            seen = _shown(sim, deployment,
                          since=now - pd.Timedelta(hours=cfg.scan_reshow_hours))
            t = scan_table(cfg, exclude=seen, now=now)
            typer.echo(scan_text(t, now) if len(t) else "no markets pass the scan filters now")
            return
        t = market_table(market, sim, cfg, series=series, horizon=pd.Timedelta(hours=float(h)),
                         late=loop.late_tolerance, now=now)
        typer.echo(opening_text(t, now) if len(t) else "no markets at their decision point now")
        return
    try:
        load_deployment(sim, deployment)
    except DeploymentError:
        # Hourly job runs this before the owner has opted in: nothing to do, not an error.
        typer.echo(f"{deployment} not registered; run `python -m desk.cli register` to start")
        return
    out = run_desk_cycle(market, sim, cfg, loop,
                         review=[t.strip() for t in review.split(",") if t.strip()] or None)
    typer.echo(json.dumps({"desk": out["desk"], "gate": out["gate"]}, indent=2, default=str))


if __name__ == "__main__":
    app()
