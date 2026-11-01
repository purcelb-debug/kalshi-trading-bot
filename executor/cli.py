"""executor CLI. Default mode is dry_run; demo/live need explicit configuration in executor/.env."""

from pathlib import Path

import typer

from executor.auth import load_private_key
from executor.client import KalshiTradingClient
from executor.config import ExecutorSettings
from executor.gate import load_gate
from executor.runner import run_cycle
from executor.safety import AuditLog, State, kill_switch_on, startup_checks
from predlab.db.repo import init_sim_db, make_engine

app = typer.Typer(add_completion=False, help="Kalshi order executor (dry_run | demo | live).")


def _client(cfg: ExecutorSettings) -> KalshiTradingClient | None:
    if cfg.mode == "dry_run":
        return None
    key_id, key_path = cfg.credentials()
    return KalshiTradingClient(cfg.base_url, key_id, load_private_key(key_path))


def _banner(cfg):
    color = {"dry_run": typer.colors.GREEN, "demo": typer.colors.YELLOW,
             "live": typer.colors.RED}[cfg.mode]
    label = {"dry_run": "DRY RUN - no orders are sent",
             "demo": "DEMO EXCHANGE - mock funds",
             "live": "LIVE - REAL MONEY"}[cfg.mode]
    typer.secho(f"[executor] {label}", fg=color, bold=True, err=True)
    if cfg.gate_overridden:
        typer.secho("[executor] GATE OVERRIDDEN: trading a strategy with no validated edge",
                    fg=typer.colors.RED, bold=True, err=True)


@app.command()
def status(model_version: str = typer.Option("", help="show go-live gate for this model")):
    """Mode, limits, kill switch, halt state, gate, and (demo/live) account balance."""
    cfg = ExecutorSettings()
    _banner(cfg)
    state = State.load(Path(cfg.data_dir) / f"state_{cfg.mode}.json")
    typer.echo(f"mode={cfg.mode}  base_url={cfg.base_url}")
    typer.echo(f"limits: bankroll_cap={cfg.bankroll_cap_usd} max_order={cfg.max_order_usd} "
               f"daily_loss_stop={cfg.daily_loss_stop_usd} max_market={cfg.max_market_usd} "
               f"max_orders/day={cfg.max_orders_per_day}")
    typer.echo(f"kill_switch={'ON' if kill_switch_on(cfg) else 'off'}  "
               f"halted={state.halted or 'no'}  orders_today={state.orders_today}")
    chk = startup_checks(cfg, state)
    typer.echo(f"ready={chk.ok} {chk.reason}")
    if model_version:
        sim = make_engine(cfg.sim_db_url)
        init_sim_db(sim)
        g = load_gate(sim, model_version, cfg.gate_paper_weeks, cfg.gate_min_paper_trades)
        typer.echo(f"gate[{model_version}] open={g.open}")
        for r in g.reasons:
            typer.echo(f"  - {r}")
    audit = AuditLog(Path(cfg.data_dir) / f"audit_{cfg.mode}.jsonl")
    typer.echo(f"audit log intact={audit.verify()} ({audit.path})")
    if chk.ok and cfg.mode != "dry_run":
        c = _client(cfg)
        try:
            cash, pv = c.balance_usd()
            typer.echo(f"account: cash=${cash:.2f} positions_value=${pv:.2f} "
                       f"open_positions={len(c.positions())}")
        finally:
            c.close()


@app.command()
def run(model_version: str = typer.Option(...), forecast_max_age_h: float = 6.0):
    """Run one cycle for a model's latest forecasts."""
    cfg = ExecutorSettings()
    _banner(cfg)
    sim = make_engine(cfg.sim_db_url)
    init_sim_db(sim)
    client = None
    try:
        client = _client(cfg) if startup_checks(cfg, State.load(
            Path(cfg.data_dir) / f"state_{cfg.mode}.json")).ok else None

        def confirm(plan):
            return typer.confirm(
                f"LIVE ORDER: buy {plan.count} {plan.outcome.upper()} {plan.ticker} "
                f"at <= ${plan.max_price:.2f} (best ask {plan.best_ask:.2f}, est. cost "
                f"${plan.cost_usd:.2f}, model p={plan.p_side:.3f}). Send?", default=False)

        for r in run_cycle(cfg, sim, model_version, client=client, confirm=confirm,
                           forecast_max_age_h=forecast_max_age_h):
            typer.echo(r)
    finally:
        if client is not None:
            client.close()


@app.command()
def sync():
    """Copy fills, settlements and a balance snapshot from Kalshi into the local ledger."""
    from executor.ledger import sync as do_sync
    cfg = ExecutorSettings()
    if cfg.mode == "dry_run":
        typer.echo("sync: skipped in dry_run (no account)")
        return
    c = _client(cfg)
    try:
        typer.echo(f"sync: {do_sync(c, cfg.data_dir)}")
    finally:
        c.close()


@app.command()
def kill():
    """Create the kill switch file. No order is sent while it exists."""
    cfg = ExecutorSettings()
    Path(cfg.kill_switch_path).write_text("kill switch engaged\n", encoding="utf-8")
    AuditLog(Path(cfg.data_dir) / f"audit_{cfg.mode}.jsonl").write("kill_switch_on")
    typer.secho(f"kill switch ON: {cfg.kill_switch_path}", fg=typer.colors.RED, bold=True)


@app.command()
def unkill():
    """Remove the kill switch file."""
    cfg = ExecutorSettings()
    Path(cfg.kill_switch_path).unlink(missing_ok=True)
    AuditLog(Path(cfg.data_dir) / f"audit_{cfg.mode}.jsonl").write("kill_switch_off")
    typer.echo("kill switch off")


@app.command()
def resume():
    """Clear a halt (daily loss stop or repeated errors) after you have reviewed it."""
    cfg = ExecutorSettings()
    p = Path(cfg.data_dir) / f"state_{cfg.mode}.json"
    s = State.load(p)
    reason, s.halted, s.consecutive_errors = s.halted, "", 0
    s.save(p)
    AuditLog(Path(cfg.data_dir) / f"audit_{cfg.mode}.jsonl").write("resume", cleared=reason)
    typer.echo(f"cleared halt: {reason or '(none)'}")


@app.command("demo-test-order")
def demo_test_order(ticker: str = typer.Option(...)):
    """DEMO ONLY: send one 1-contract IOC buy at $0.01 to verify signing and order flow."""
    cfg = ExecutorSettings()
    _banner(cfg)
    if cfg.mode != "demo":
        raise typer.BadParameter("demo-test-order only runs with EXECUTOR_MODE=demo")
    chk = startup_checks(cfg, State.load(Path(cfg.data_dir) / "state_demo.json"))
    if not chk.ok:
        raise typer.Exit(f"not ready: {chk.reason}")
    c = _client(cfg)
    audit = AuditLog(Path(cfg.data_dir) / "audit_demo.jsonl")
    try:
        audit.write("demo_test_submit", ticker=ticker)
        ack = c.place_ioc_buy(ticker, "yes", 1, 0.01)
        audit.write("demo_test_ack", **ack.__dict__)
        typer.echo(ack)
        typer.echo(f"balance: {c.balance_usd()}")
    finally:
        c.close()


@app.command("live-test-order")
def live_test_order(ticker: str = typer.Option(...),
                    side: str = typer.Option("yes", help="yes | no"),
                    confirm_real_money: bool = typer.Option(
                        False, "--confirm-real-money", help="required: this uses your money")):
    """LIVE: one 1-contract IOC buy (YES or NO) at $0.01 to verify signing and order flow on the
    real exchange. Usually does not fill; worst case it costs $0.01 plus fee."""
    cfg = ExecutorSettings()
    _banner(cfg)
    if cfg.mode != "live":
        raise typer.BadParameter("live-test-order only runs with EXECUTOR_MODE=live")
    if not confirm_real_money:
        raise typer.BadParameter("pass --confirm-real-money to send a real order")
    if side not in ("yes", "no"):
        raise typer.BadParameter("side must be yes or no")
    chk = startup_checks(cfg, State.load(Path(cfg.data_dir) / "state_live.json"))
    if not chk.ok:
        raise typer.Exit(f"not ready: {chk.reason}")
    c = _client(cfg)
    audit = AuditLog(Path(cfg.data_dir) / "audit_live.jsonl")
    try:
        cash, pv = c.balance_usd()
        typer.echo(f"balance before: cash=${cash:.2f} positions=${pv:.2f}")
        audit.write("live_test_submit", ticker=ticker, side=side)
        ack = c.place_ioc_buy(ticker, side, 1, 0.01)
        audit.write("live_test_ack", **ack.__dict__)
        typer.echo(ack)
        cash, pv = c.balance_usd()
        typer.echo(f"balance after:  cash=${cash:.2f} positions=${pv:.2f}")
    finally:
        c.close()


if __name__ == "__main__":
    app()
