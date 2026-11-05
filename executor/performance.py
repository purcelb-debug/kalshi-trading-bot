"""Offline attribution from filled orders, keeping manual cash flows out of bot results."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from executor.ledger import _bot_order_ids

# First match wins: election tickers must precede precipitation tickers.
CATEGORIES = [
    ("politics/approval", ("KXRAINELECTION", "KXTRUMP", "KXAPRPOTUS")),
    ("weather_temp", ("KXHIGH", "KXLOWT")),
    ("rain", ("KXRAIN",)),
    ("crypto", ("KXBTC", "KXETH", "KXSOL", "KXXRP", "KXDOGE")),
    ("energy", ("KXWTI", "KXBRENT", "KXNATGAS", "KXDIESEL", "KXAAAGAS")),
    ("metals", ("KXGOLD", "KXSILVER", "KXCOPPER")),
    ("indices", ("KXINX", "KXNASDAQ100", "KXDJI")),
    ("rates", ("KXUST", "KX30YMORT")),
    ("other", ("",)),
]


def category(ticker: str) -> str:
    """Use series prefixes so strike and date text cannot change the category."""
    series = ticker.split("-", 1)[0]
    return next(name for name, prefixes in CATEGORIES if series.startswith(prefixes))


def order_context(audit: pd.DataFrame) -> dict:
    """Pair acknowledgements with their submit, retaining the cycle deployment."""
    orders, pending, model = {}, {}, None
    for event in audit.to_dict("records"):
        kind = event.get("event")
        if kind == "cycle_start":
            model = event.get("model_version")
            model = None if pd.isna(model) else model
        elif kind in ("order_submit", "exit_submit"):
            pending[kind] = {"plan": event.get("plan") or {}, "model": model,
                             "ts": event.get("ts")}
        elif kind in ("order_ack", "exit_ack") and event.get("order_id"):
            submit = pending.pop("exit_submit" if kind == "exit_ack" else "order_submit", {})
            orders[event["order_id"]] = {**submit, "exit": kind == "exit_ack"}
    return orders


def _explanation(value):
    if isinstance(value, str):
        return json.loads(value)
    return value if isinstance(value, dict) else {}


def _forecast(forecasts, ticker, ts, model, side, probability):
    candidates = forecasts.loc[(forecasts["ticker"] == ticker)
                               & (forecasts["created_ts"] <= ts)]
    if model:
        candidates = candidates.loc[candidates["model_version"] == model]
    # Without an audited deployment, matching p_side avoids choosing an unrelated model.
    if candidates.empty:
        return model or "unknown", probability
    latest = candidates.sort_values("created_ts").groupby("model_version").tail(1)
    latest = latest.loc[~latest["explanation"].map(lambda e: _explanation(e).get(
        "withdrawn", False)).astype(bool)]
    if probability is not None and not latest.empty:
        p_yes = probability if side == "yes" else 1 - probability
        latest = latest.loc[np.isclose(latest["p"], p_yes, atol=1e-8)]
    if len(latest) != 1:
        return model or "unknown", probability
    row = latest.iloc[0]
    return row.model_version, (probability if probability is not None
                               else float(row.p if side == "yes" else 1 - row.p))


def trade_records(fills, settlements, audit, forecasts, bot_order_ids=None):
    """Attribute FIFO entry lots; never use account-wide settlement cost as bot cost."""
    orders = order_context(audit)
    bot_ids = set(orders) if bot_order_ids is None else bot_order_ids
    fc = forecasts.copy()
    if fc.empty:
        fc = pd.DataFrame(columns=["ticker", "created_ts", "model_version", "p", "explanation"])
    fc["created_ts"] = pd.to_datetime(fc["created_ts"], utc=True)
    lots, closed, warnings = [], [], []
    for fill in (fills.sort_values("ts").to_dict("records") if not fills.empty else []):
        oid, ticker = fill["order_id"], fill["ticker"]
        ctx = orders.get(oid, {})
        plan = ctx.get("plan", {})
        source = "bot" if oid in bot_ids and not plan.get("manual") else "manual"
        side, price = fill["outcome"], float(fill["price"])
        intended = plan.get("held" if ctx.get("exit") else "outcome")
        # ledger.norm_fill already selects outcome_side's price. Complement only when
        # the audited contract actually differs; action=sell alone is not an exit.
        if intended in ("yes", "no") and intended != side:
            side, price = intended, 1 - price
        count, fee = float(fill["count"]), float(fill["fee"])
        if count <= 0:
            continue
        ts = pd.to_datetime(fill["ts"], utc=True)
        candidates = [lot for lot in lots if lot["ticker"] == ticker
                      and lot["outcome"] == side and lot["source"] == source
                      and lot["remaining"] > 0]
        is_exit = ctx.get("exit", False) or (
            not ctx and fill["action"] == "sell" and bool(candidates))
        if is_exit:
            remaining = count
            for lot in candidates:
                quantity = min(remaining, lot["remaining"])
                if quantity <= 0:
                    break
                entry_cost = quantity * lot["price"]
                fees = quantity * (lot["fee_per_contract"] + fee / count)
                closed.append({**lot, "count": quantity, "cost": entry_cost, "fees": fees,
                               "pnl": quantity * price - entry_cost - fees,
                               "closed_ts": ts, "kind": "exit", "target": None})
                lot["remaining"] -= quantity
                remaining -= quantity
            if remaining > 1e-8:
                warnings.append(f"Unmatched exit: {oid} ({remaining:g} contracts)")
            continue
        model, probability = _forecast(fc, ticker, pd.to_datetime(ctx.get("ts") or ts,
                                       utc=True), ctx.get("model"), side, plan.get("p_side"))
        lots.append({"ticker": ticker, "order_id": oid, "category": category(ticker),
                     "source": source, "deployment": model if source == "bot" else "manual",
                     "outcome": side, "price": price, "probability": probability,
                     "remaining": count, "fee_per_contract": fee / count, "entry_ts": ts})
    for settlement in settlements.sort_values("settled_ts").to_dict("records"):
        ts = pd.to_datetime(settlement["settled_ts"], utc=True)
        held = [lot for lot in lots if lot["ticker"] == settlement["ticker"]
                and lot["entry_ts"] <= ts and lot["remaining"] > 0]
        if not held:
            continue
        total = sum(lot["remaining"] for lot in held)
        market_fees = fills.loc[fills["ticker"] == settlement["ticker"], "fee"].sum()
        # Ledger settle_fee includes trading fees; subtract them only once.
        extra_fee = max(0.0, float(settlement["settle_fee"]) - float(market_fees))
        for lot in held:
            quantity = lot["remaining"]
            result = settlement["result"]
            if result not in ("yes", "no"):
                warnings.append(f"Unknown settlement result: {settlement['ticker']} {result}")
                continue
            target = float(result == lot["outcome"])
            cost = quantity * lot["price"]
            fees = quantity * lot["fee_per_contract"] + extra_fee * quantity / total
            closed.append({**lot, "count": quantity, "cost": cost, "fees": fees,
                           "pnl": quantity * target - cost - fees,
                           "closed_ts": ts, "kind": "settled", "target": target})
            lot["remaining"] = 0
    return closed, [lot for lot in lots if lot["remaining"] > 1e-8], sorted(set(warnings))


def _metrics(records):
    # Multiple fills of one entry order are one trade, including partial fills.
    frame = pd.DataFrame(records)
    grouped = frame.groupby("order_id", sort=True)
    pnl = grouped["pnl"].sum().to_numpy()
    fees, cost = float(frame["fees"].sum()), float(frame["cost"].sum())
    rng = np.random.default_rng(42)
    means = [float(rng.choice(pnl, size=len(pnl), replace=True).mean()) for _ in range(2000)]
    market = frame.loc[frame["target"].notna()].copy()
    model_errors, market_errors = [], []
    for _, entry in market.groupby("order_id"):
        weights = entry["count"]
        target = float(entry["target"].iloc[0])
        entry_price = float(np.average(entry["price"], weights=weights))
        market_errors.append((entry_price - target) ** 2)
        if entry["probability"].notna().all():
            probability = float(np.average(entry["probability"], weights=weights))
            model_errors.append((probability - target) ** 2)
    return {"trades": len(pnl), "win_rate": float((pnl > 0).mean()),
            "net_pnl": float(pnl.sum()), "pnl_per_trade": float(pnl.mean()), "fees": fees,
            "capital_deployed": cost, "return_on_deployed_pct": 100 * pnl.sum() / cost
            if cost else None, "average_entry_price": cost / float(frame["count"].sum()),
            "brier_bot": float(np.mean(model_errors)) if model_errors else None,
            "brier_market": float(np.mean(market_errors)) if market_errors else None,
            "brier_trades": len(model_errors),
            "pnl_per_trade_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "sample": "insufficient sample" if len(pnl) < 50 else "sufficient sample"}


def build_report(fills, settlements, audit, forecasts, days=None, now=None, bot_order_ids=None):
    """Filter by resolution time, preserving older entries needed for their cost basis."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    if days is not None and days < 1:
        raise ValueError("days must be positive")
    if not fills.empty:
        fills = fills.loc[pd.to_datetime(fills["ts"], utc=True) <= now]
    if not settlements.empty:
        settlements = settlements.loc[pd.to_datetime(settlements["settled_ts"], utc=True) <= now]
    closed, opened, warnings = trade_records(fills, settlements, audit, forecasts, bot_order_ids)
    closed = [row for row in closed if row["closed_ts"] <= now and (days is None or
              row["closed_ts"] >= now - pd.Timedelta(days=days))]
    report = {"as_of": now.isoformat(), "days": days, "warnings": warnings}
    for source in ("bot", "manual"):
        selected = [r for r in closed if r["source"] == source]
        sections = {}
        for kind in ("settled", "exit"):
            records = [r for r in selected if r["kind"] == kind]
            sections[kind] = {}
            for field in ("category", "deployment"):
                sections[kind][f"by_{field}"] = [
                    {field: key, **_metrics([r for r in records if r[field] == key])}
                    for key in sorted({r[field] for r in records})]
        sections["open_by_category"] = [
            {"category": key, "positions": len({(r["ticker"], r["outcome"]) for r in opened
             if r["source"] == source and r["category"] == key}),
             "contracts": sum(r["remaining"] for r in opened
                              if r["source"] == source and r["category"] == key),
             "cost_basis": sum(r["remaining"] * (r["price"] + r["fee_per_contract"])
                               for r in opened if r["source"] == source and r["category"] == key)}
            for key in sorted({r["category"] for r in opened if r["source"] == source})]
        report[source] = sections
    report["methodology"] = (
        "Trade = filled entry order; exits and settlements shown separately. FIFO within owner, "
        "ticker and outcome. Win = positive net P&L. Brier = mean squared error per entry order "
        "on settled contracts; missing probabilities excluded and coverage reported. "
        "Capital excludes fees; open cost basis includes entry fees. Settlement payouts use "
        "contract outcome, never account-wide cost attribution. Unattributed deployments stay "
        "unknown. Days filters resolution time; open positions show the full current inventory. "
        "CI: 2000 entry-order bootstrap resamples, seed 42; correlated trades may understate risk."
    )
    return report


def load_report(data_dir="executor/data", sim_path="data/sim.db", days=None):
    """SQLite mode=ro avoids the schema creation and writes in ledger.load."""
    def read(path, table):
        with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)

    data_dir = Path(data_dir)
    audit = pd.DataFrame([json.loads(line) for line in
                          (data_dir / "audit_live.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()])
    return build_report(read(data_dir / "ledger.db", "fills"),
                        read(data_dir / "ledger.db", "settlements"), audit,
                        read(sim_path, "forecasts"), days=days,
                        bot_order_ids=_bot_order_ids(str(data_dir)))


def format_report(report):
    """Plain text stays readable offline and when redirected to a file."""
    lines = [f"Performance as of {report['as_of']} (days={report['days'] or 'all'})"]
    for source in ("bot", "manual"):
        for kind in ("settled", "exit"):
            for field in ("category", "deployment"):
                lines.append(f"\n{source.upper()} {kind} by {field}")
                lines.append("Group | Trades | Win% | Net$ | $/trade | Fees$ | Capital$ | ROI% | "
                             "Entry$ | Brier bot/market (n) | 95% CI $/trade | Sample")
                for row in report[source][kind][f"by_{field}"]:
                    def fmt(value):
                        return "n/a" if value is None else f"{value:.4f}"

                    lo, hi = row["pnl_per_trade_ci95"]
                    lines.append(
                        f"{row[field]} | {row['trades']} | {100 * row['win_rate']:.1f} | "
                        f"{row['net_pnl']:.2f} | {row['pnl_per_trade']:.2f} | {row['fees']:.2f} | "
                        f"{row['capital_deployed']:.2f} | {fmt(row['return_on_deployed_pct'])} | "
                        f"{row['average_entry_price']:.3f} | {fmt(row['brier_bot'])}/"
                        f"{fmt(row['brier_market'])} ({row['brier_trades']}) | "
                        f"[{lo:.2f}, {hi:.2f}] | {row['sample']}")
                if not report[source][kind][f"by_{field}"]:
                    lines.append("(none)")
        lines.append(f"\n{source.upper()} open positions (including entry fees)")
        for row in report[source]["open_by_category"]:
            lines.append(f"{row['category']}: {row['positions']} positions, "
                         f"{row['contracts']:g} contracts, cost basis ${row['cost_basis']:.2f}")
    lines.extend(["", report["methodology"], *report["warnings"]])
    return "\n".join(lines)
