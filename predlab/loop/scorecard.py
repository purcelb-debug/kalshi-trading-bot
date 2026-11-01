"""Weekly scorecard for a deployment's forward paper record. PAPER TRADING ONLY.

Reads the persisted ``paper-<deployment>`` sim run (rebuilt every loop cycle) and reports
the trading statistics a desk should be graded on: win rate, average win and loss,
expectancy per trade, worst losing streak, P&L, fees and drawdown, plus the forward Brier
comparison and the raw gate evidence. Pure reads; changes nothing.
"""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from predlab.db.schema import model_registry, sim_positions, sim_runs
from predlab.loop.runner import _resolutions, paper_run_id
from predlab.timeutil import utcnow


def trade_stats(trades: pd.DataFrame) -> dict:
    """Statistics over settled trades with columns ``net`` and ``resolve_ts``."""
    t = trades.sort_values("resolve_ts")
    n = len(t)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "avg_win": None,
                "avg_loss": None, "expectancy": None, "worst_losing_streak": 0, "net_pnl": 0.0}
    wins, losses = t[t["net"] > 0]["net"], t[t["net"] <= 0]["net"]
    streak = worst = 0
    for net in t["net"]:
        streak = streak + 1 if net <= 0 else 0
        worst = max(worst, streak)
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        # Equals win_rate * avg_win + loss_rate * avg_loss (avg_loss is negative).
        "expectancy": float(t["net"].mean()),
        "worst_losing_streak": worst,
        "net_pnl": round(float(t["net"].sum()), 4),
    }


def scorecard(market: Engine, sim: Engine, deployment: str, *, days: int | None = 7,
              now=None) -> dict:
    """Scorecard over trades settled in the last ``days`` days (None = whole record)."""
    now = pd.Timestamp(now if now is not None else utcnow())
    run_id = paper_run_id(deployment)
    with sim.connect() as c:
        pos = pd.DataFrame(c.execute(select(sim_positions).where(
            sim_positions.c.run_id == run_id, sim_positions.c.settled.is_(True))).mappings().all(),
            columns=[col.name for col in sim_positions.c])
        run = c.execute(select(sim_runs.c.summary_json).where(sim_runs.c.run_id == run_id)).scalar()
        reg = c.execute(select(model_registry.c.status, model_registry.c.validation_json)
                        .where(model_registry.c.model_version == deployment)).first()
    if len(pos):
        res = _resolutions(market, pos["ticker"].unique().tolist(), now)
        pos = pos.merge(res[["ticker", "resolve_ts"]], on="ticker", how="left")
        pos["net"] = pos["payout"].fillna(0.0) - pos["cost_basis"] - pos["fees_paid"]
    else:
        pos = pd.DataFrame(columns=["ticker", "net", "resolve_ts"])
    window = pos
    if days is not None and len(pos):
        window = pos[pd.to_datetime(pos["resolve_ts"], utc=True) >= now - timedelta(days=days)]
    summary = run or {}
    gate = ((reg.validation_json if reg else None) or {}).get("gate", {})
    return {
        "deployment": deployment,
        "registry_status": reg.status if reg else None,
        "as_of": str(now),
        "window_days": days,
        "window": trade_stats(window),
        "all_time": trade_stats(pos),
        "paper": {k: summary.get(k) for k in ("net_pnl", "fees", "max_drawdown", "n_fills",
                                              "open_positions_at_end", "net_pnl_ci95")},
        "brier_diff_vs_market": gate.get("paper_brier_diff_vs_market"),
        "gate_evidence": gate,
    }


def write_scorecard(card: dict, report_dir: str = "reports") -> Path:
    d = Path(report_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = card["deployment"].replace("@", "_").replace("/", "_")
    path = d / f"scorecard-{safe}-{pd.Timestamp(card['as_of']):%Y-%m-%d}.json"
    path.write_text(json.dumps(card, indent=2, default=str))
    return path


def format_scorecard(card: dict) -> str:
    def f(v, spec):
        return "n/a" if v is None else format(v, spec)

    lines = [f"{card['deployment']} [{card['registry_status']}]  as of {card['as_of'][:16]}"]
    for label, s in ((f"last {card['window_days']}d" if card["window_days"] else "window",
                      card["window"]), ("all time", card["all_time"])):
        lines.append(f"  {label}: {s['trades']} trades, win rate {f(s['win_rate'], '.0%')}, "
                     f"avg win {f(s['avg_win'], '.2f')}, avg loss {f(s['avg_loss'], '.2f')}, "
                     f"expectancy {f(s['expectancy'], '.3f')}/trade, "
                     f"worst losing streak {s['worst_losing_streak']}, net {s['net_pnl']:.2f}")
    p = card["paper"]
    lines.append(f"  paper: net {p.get('net_pnl')}  fees {p.get('fees')}  "
                 f"max drawdown {f(p.get('max_drawdown'), '.1%')}  "
                 f"open {p.get('open_positions_at_end')}")
    lines.append(f"  Brier vs market (negative = better): {card['brier_diff_vs_market']}")
    return "\n".join(lines)
