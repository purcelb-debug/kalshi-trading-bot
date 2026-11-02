"""notify command line: ``digest`` (run after every loop cycle) and ``weekly`` (Fridays)."""

import sys

import typer

from notify.config import NotifySettings
from notify.digest import build_messages, load_state, save_state
from notify.telegram import send

app = typer.Typer(add_completion=False, help="Outbound-only phone notifications.")


@app.callback()
def _main():
    # The scheduled job logs to a file in the Windows code page; LLM transcripts contain
    # characters it can't encode, and a crash here would block every later message.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _deliver(cfg: NotifySettings, messages: list[str], dry: bool) -> bool:
    ok = True
    for m in messages:
        typer.echo(m + "\n")
        if not dry:
            sent, err = send(cfg, m)
            if not sent and err != "disabled":
                typer.echo(err, err=True)
                ok = False
    return ok


@app.command()
def digest(dry: bool = typer.Option(False, "--dry", help="print only, send nothing"),
           test: bool = typer.Option(False, "--test", help="send one test message")):
    """Send what changed since the last run: alerts, gate changes, executor events,
    desk transcripts, and a warning when the loop has gone quiet."""
    cfg = NotifySettings()
    if test:
        sent, err = send(cfg, "predlab notifier test: Telegram is wired up. PAPER TRADING.")
        typer.echo("delivered" if sent else f"not delivered: {err}")
        raise typer.Exit(0 if sent else 1)
    state = load_state(cfg.state_path)
    messages = build_messages(cfg, state)
    # Keep progress only when delivery worked (or the notifier is off), so a Telegram
    # outage resends next time; dry runs never advance it.
    if _deliver(cfg, messages, dry) and not dry:
        save_state(cfg.state_path, state)


@app.command()
def weekly(days: int = 7, dry: bool = typer.Option(False, "--dry")):
    """The Friday grade: scorecard plus go-live gate verdict for every live deployment."""
    from sqlalchemy import select

    from executor.config import ExecutorSettings
    from executor.gate import evaluate_gate
    from predlab.config import get_settings
    from predlab.db.repo import init_sim_db, make_engine
    from predlab.db.schema import model_registry
    from predlab.loop.scorecard import format_scorecard, scorecard, write_scorecard

    cfg, ex, s = NotifySettings(), ExecutorSettings(), get_settings()
    market, sim = make_engine(s.market_db_url), make_engine(s.sim_db_url)
    init_sim_db(sim)
    with sim.connect() as c:
        deps = [r.model_version for r in c.execute(
            select(model_registry.c.model_version).where(model_registry.c.status != "retired"))]
    parts = []
    for d in deps:
        card = scorecard(market, sim, d, days=days)
        write_scorecard(card, cfg.report_dir)
        g = evaluate_gate(d, card["gate_evidence"], ex.gate_paper_weeks, ex.gate_min_paper_trades)
        parts.append(format_scorecard(card) + f"\n  go-live gate: {'OPEN' if g.open else 'closed'}"
                     + "".join(f"\n    - {r}" for r in g.reasons))
    text = "[WEEKLY SCORECARD] PAPER TRADING\n\n" + ("\n\n".join(parts) or "no deployments")
    sys.exit(0 if _deliver(cfg, [text], dry) else 1)


if __name__ == "__main__":
    app()
