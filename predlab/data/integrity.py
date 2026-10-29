"""Integrity checks on normalized rows. Each check returns issue dicts; nothing is dropped
silently. Rows with hard errors are excluded by the caller and the issue is recorded."""

from datetime import timedelta

from predlab.timeutil import utcnow

FUTURE_TOLERANCE = timedelta(days=1)
SETTLED_STATUSES = {"settled", "finalized"}


def _issue(source, entity, entity_id, check, detail, hard=False):
    return {"source": source, "entity": entity, "entity_id": entity_id,
            "check_name": check, "detail": detail, "_hard": hard}


def _prob_ok(x) -> bool:
    return x is None or 0.0 <= x <= 1.0


def check_market(row: dict) -> list[dict]:
    src, tid = row["source"], row["ticker"]
    out = []
    for col in ("yes_bid", "yes_ask", "no_bid", "no_ask", "last_price"):
        if not _prob_ok(row.get(col)):
            out.append(_issue(src, "market", tid, f"{col}_range", f"{col}={row.get(col)}", hard=True))
    yb, ya = row.get("yes_bid"), row.get("yes_ask")
    if yb is not None and ya is not None and ya > 0 and yb > ya:
        out.append(_issue(src, "market", tid, "crossed_book", f"yes_bid {yb} > yes_ask {ya}"))
    ot, ct = row.get("open_time"), row.get("close_time")
    if ot and ct and ct < ot:
        out.append(_issue(src, "market", tid, "close_before_open", f"{ot} > {ct}", hard=True))
    status, result = (row.get("status") or "").lower(), (row.get("result") or "").lower()
    if status in SETTLED_STATUSES and result not in ("yes", "no"):
        out.append(_issue(src, "market", tid, "settled_without_binary_result",
                          f"status={status} result={result!r}"))
    if result in ("yes", "no") and status not in SETTLED_STATUSES | {"closed", "determined"}:
        out.append(_issue(src, "market", tid, "result_on_open_market", f"status={status}"))
    return out


def check_trade(row: dict, source: str = "kalshi") -> list[dict]:
    tid = row["trade_id"]
    out = []
    if not (0.0 < row["yes_price"] < 1.0):
        out.append(_issue(source, "trade", tid, "price_range", f"yes_price={row['yes_price']}", hard=True))
    if row["count"] <= 0:
        out.append(_issue(source, "trade", tid, "nonpositive_count", f"count={row['count']}", hard=True))
    if row.get("no_price") is not None and abs(row["yes_price"] + row["no_price"] - 1.0) > 1e-6:
        out.append(_issue(source, "trade", tid, "yes_no_not_complementary",
                          f"{row['yes_price']}+{row['no_price']}"))
    if row["created_time"] > utcnow() + FUTURE_TOLERANCE:
        out.append(_issue(source, "trade", tid, "future_timestamp", str(row["created_time"]), hard=True))
    return out


def check_candle(row: dict, source: str = "kalshi") -> list[dict]:
    key = f"{row['ticker']}@{row['end_ts'].isoformat()}"
    out = []
    lo, hi, cl = row.get("price_low"), row.get("price_high"), row.get("price_close")
    if None not in (lo, hi, cl) and not (lo - 1e-9 <= cl <= hi + 1e-9):
        out.append(_issue(source, "candle", key, "ohlc_inconsistent", f"l={lo} c={cl} h={hi}", hard=True))
    for col in ("price_close", "yes_bid_close", "yes_ask_close"):
        if not _prob_ok(row.get(col)):
            out.append(_issue(source, "candle", key, f"{col}_range", str(row.get(col)), hard=True))
        elif row.get(col) is not None and 0 < row[col] < 0.01:
            # Kalshi's standard tick is 1 cent; a sub-cent price usually means a unit error.
            out.append(_issue(source, "candle", key, "sub_cent_price", f"{col}={row[col]}"))
    return out


def check_observation(row: dict) -> list[dict]:
    key = f"{row['series_id']}@{row['obs_ts'].isoformat()}"
    out = []
    if row.get("value") is None:
        out.append(_issue(row["source"], "observation", key, "missing_value", "value is null"))
    if row["vintage_ts"] < row["obs_ts"] - timedelta(days=1):
        out.append(_issue(row["source"], "observation", key, "vintage_before_observation",
                          f"vintage {row['vintage_ts']} < obs {row['obs_ts']}", hard=True))
    return out


def split_valid(rows: list[dict], checker) -> tuple[list[dict], list[dict]]:
    """Return (rows without hard errors, all issues with the internal flag stripped)."""
    good, issues = [], []
    for r in rows:
        found = checker(r)
        if not any(i["_hard"] for i in found):
            good.append(r)
        issues.extend({k: v for k, v in i.items() if k != "_hard"} for i in found)
    return good, issues
