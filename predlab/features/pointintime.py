"""Point-in-time dataset construction.

For each settled binary market a *decision time* is chosen (``close_time - horizon``).
Every input is filtered to what was publicly knowable at that moment:

* candles: ``end_ts <= decision_ts`` (a candle is only known once its period ends)
* external observations: ``vintage_ts <= decision_ts`` (first-release values, not revisions)

``input_max_ts`` (the latest timestamp actually used) is recorded per row and checked
against the decision time, so a leak raises instead of silently inflating skill.

Label availability (``label_ts``) is the market's expiration time, which is conservative
(Kalshi's ``expiration_time`` is the latest possible settlement). The splitter only trains
on rows whose label was known before the validation window starts.
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, and_, func, select

from predlab.db.schema import ext_observations, mkt_candles, mkt_markets, outcomes

MARKET_FEATURES = ("mkt_mid", "mkt_spread", "mkt_log_volume", "mkt_stale_hours",
                   "mkt_mom6", "mkt_mom24")
MOMENTUM_LAGS_H = (6, 24)
# Kalshi high-temperature series -> station key used by the weather collector (assumption).
WEATHER_SERIES_STATION = {"KXHIGHNY": "NYC", "KXHIGHCHI": "CHI", "KXHIGHMIA": "MIA"}
PER_MARKET_SOURCES = ("synthetic", "csv_sports")
STALE_HOURS = 6.0
WIDE_SPREAD = 0.10
_EVENT_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")
_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


class LookAheadError(AssertionError):
    pass


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "Dataset":
        mask = np.asarray(mask)
        return Dataset(self.X[mask].reset_index(drop=True), self.y[mask],
                       self.meta[mask].reset_index(drop=True), list(self.feature_names))


def _to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def load_markets(engine: Engine, *, source=None, category=None, series=None,
                 as_of=None) -> pd.DataFrame:
    """Settled binary markets with their latest yes/no outcome and revision flag.
    With ``as_of``, only outcomes recorded by then count (replay / forward use)."""
    lq = select(outcomes.c.ticker, func.max(outcomes.c.recorded_at).label("rec"),
                func.max(outcomes.c.revised).label("revised"))
    if as_of is not None:
        lq = lq.where(outcomes.c.recorded_at <= pd.Timestamp(as_of).to_pydatetime())
    latest = lq.group_by(outcomes.c.ticker).subquery()
    q = (select(mkt_markets.c.ticker, mkt_markets.c.source, mkt_markets.c.category,
                mkt_markets.c.series_ticker, mkt_markets.c.event_ticker, mkt_markets.c.open_time,
                mkt_markets.c.close_time, mkt_markets.c.expiration_time,
                mkt_markets.c.strike_type, mkt_markets.c.floor_strike, mkt_markets.c.cap_strike,
                outcomes.c.result, latest.c.revised)
         .join(latest, latest.c.ticker == mkt_markets.c.ticker)
         .join(outcomes, and_(outcomes.c.ticker == latest.c.ticker,
                              outcomes.c.recorded_at == latest.c.rec))
         .where(outcomes.c.result.in_(("yes", "no"))))
    for col, val in ((mkt_markets.c.source, source), (mkt_markets.c.category, category),
                     (mkt_markets.c.series_ticker, series)):
        if val is not None:
            q = q.where(col.in_(val) if isinstance(val, (list, tuple, set)) else col == val)
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(q).mappings().all())
    if df.empty:
        return df
    for col in ("open_time", "close_time", "expiration_time"):
        df[col] = _to_ts(df[col])
    df["revised"] = df["revised"].fillna(False).astype(bool)
    return df


def _market_state(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """As-of join of the last *completed* hourly candle before each decision time."""
    tickers = base["ticker"].tolist()
    frames = []
    with engine.connect() as c:
        for i in range(0, len(tickers), 900):
            q = (select(mkt_candles.c.ticker, mkt_candles.c.end_ts, mkt_candles.c.yes_bid_close,
                        mkt_candles.c.yes_ask_close, mkt_candles.c.volume)
                 .where(mkt_candles.c.period_minutes == 60,
                        mkt_candles.c.ticker.in_(tickers[i:i + 900])))
            frames.append(pd.DataFrame(c.execute(q).mappings().all()))
    candles = pd.concat(frames) if frames else pd.DataFrame()
    out = base[["ticker", "decision_ts"]].copy()
    if candles.empty:
        for f in MARKET_FEATURES:
            out[f] = np.nan
        out["mkt_ts"] = pd.NaT
        return out.drop(columns="decision_ts")
    candles["end_ts"] = _to_ts(candles["end_ts"])
    candles = candles.sort_values(["ticker", "end_ts"])
    candles["cum_vol"] = candles.groupby("ticker")["volume"].cumsum()
    bid, ask = candles["yes_bid_close"], candles["yes_ask_close"]
    valid = (ask > 0) & (ask > bid) & bid.notna() & ask.notna()
    candles["mkt_mid"] = np.where(valid, (bid + ask) / 2, np.nan)
    candles["mkt_spread"] = np.where(valid, ask - bid, np.nan)
    merged = pd.merge_asof(
        out.sort_values("decision_ts"),
        candles.rename(columns={"end_ts": "mkt_ts"}).sort_values("mkt_ts")[
            ["ticker", "mkt_ts", "mkt_mid", "mkt_spread", "cum_vol"]],
        left_on="decision_ts", right_on="mkt_ts", by="ticker", direction="backward")
    merged["mkt_log_volume"] = np.log1p(merged["cum_vol"].fillna(0))
    merged["mkt_stale_hours"] = (merged["decision_ts"] - merged["mkt_ts"]).dt.total_seconds() / 3600
    # Momentum: change in logit(mid) since the last mid known `lag` hours before decision.
    # Lagged quotes are strictly older than the current one, so no look-ahead is added.
    hist = candles.dropna(subset=["mkt_mid"]).rename(columns={"end_ts": "lag_ts"})
    for lag in MOMENTUM_LAGS_H:
        left = merged[["ticker", "decision_ts"]].assign(
            lag_key=merged["decision_ts"] - pd.Timedelta(hours=lag)).sort_values("lag_key")
        old = pd.merge_asof(left, hist.sort_values("lag_ts")[["ticker", "lag_ts", "mkt_mid"]],
                            left_on="lag_key", right_on="lag_ts", by="ticker",
                            direction="backward").set_index(["ticker", "decision_ts"])["mkt_mid"]
        old = old.reindex(pd.MultiIndex.from_frame(merged[["ticker", "decision_ts"]])).to_numpy()
        merged[f"mkt_mom{lag}"] = _logit(merged["mkt_mid"].to_numpy()) - _logit(old)
    return merged.drop(columns=["cum_vol", "decision_ts"])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def _per_market_obs(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """Observations keyed ``<feature>:<ticker>`` (e.g. synthetic x1..x3, sports ref_p)."""
    tickers = base["ticker"].tolist()
    with engine.connect() as c:
        obs = pd.DataFrame(c.execute(
            select(ext_observations.c.series_id, ext_observations.c.vintage_ts,
                   ext_observations.c.value)
            .where(ext_observations.c.source.in_(PER_MARKET_SOURCES))).mappings().all())
    if obs.empty:
        return pd.DataFrame({"ticker": tickers})
    obs["feature"] = obs["series_id"].str.split(":", n=1).str[0]
    obs["ticker"] = obs["series_id"].str.split(":", n=1).str[1]
    obs["vintage_ts"] = _to_ts(obs["vintage_ts"])
    obs = obs.merge(base[["ticker", "decision_ts"]], on="ticker")
    obs = obs[obs["vintage_ts"] <= obs["decision_ts"]]          # point-in-time filter
    obs = obs.sort_values("vintage_ts").groupby(["ticker", "feature"]).tail(1)
    wide = obs.pivot(index="ticker", columns="feature", values="value")
    wide.columns = [f"obs_{c}" for c in wide.columns]
    ts = obs.groupby("ticker")["vintage_ts"].max().rename("obs_ts")
    return wide.join(ts).reset_index()


def _event_date(event_ticker: str) -> date | None:
    m = _EVENT_DATE.search(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def _margin(level, strike_type, floor, cap):
    """Signed distance of a temperature estimate from the YES region (positive = inside)."""
    if strike_type == "greater" and floor is not None:
        return level - floor
    if strike_type == "less" and cap is not None:
        return cap - level
    if strike_type == "between" and floor is not None and cap is not None:
        return min(level - floor, cap - level)
    return np.nan


def _weather(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """Margin of the YES strike vs a trailing 7-day observed-TMAX mean and vs the latest
    NWS forecast, each restricted to values public at decision time."""
    rows = []
    wx = base[base["series_ticker"].isin(WEATHER_SERIES_STATION)]
    if wx.empty:
        return pd.DataFrame({"ticker": base["ticker"]})
    stations = {WEATHER_SERIES_STATION[s] for s in wx["series_ticker"].unique()}
    ids = [f"tmax:{s}" for s in stations] + [f"fcst_high:{s}" for s in stations]
    with engine.connect() as c:
        obs = pd.DataFrame(c.execute(
            select(ext_observations.c.series_id, ext_observations.c.obs_ts,
                   ext_observations.c.vintage_ts, ext_observations.c.value)
            .where(ext_observations.c.series_id.in_(ids))).mappings().all())
    if obs.empty:
        return pd.DataFrame({"ticker": base["ticker"]})
    obs["obs_ts"], obs["vintage_ts"] = _to_ts(obs["obs_ts"]), _to_ts(obs["vintage_ts"])
    obs = obs.dropna(subset=["value"])
    by_id = {k: g.sort_values("obs_ts") for k, g in obs.groupby("series_id")}
    for r in wx.itertuples():
        st = WEATHER_SERIES_STATION[r.series_ticker]
        target = _event_date(r.event_ticker)
        row = {"ticker": r.ticker, "wx_margin_clim7": np.nan, "wx_margin_nws": np.nan,
               "wx_ts": pd.NaT}
        if target is None:
            rows.append(row)
            continue
        target_ts = pd.Timestamp(target, tz="UTC")
        used = []
        tm = by_id.get(f"tmax:{st}")
        if tm is not None:
            known = tm[(tm["vintage_ts"] <= r.decision_ts) & (tm["obs_ts"] < target_ts)].tail(7)
            if len(known) >= 3:
                row["wx_margin_clim7"] = _margin(known["value"].mean(), r.strike_type,
                                                 r.floor_strike, r.cap_strike)
                used.append(known["vintage_ts"].max())
        fc = by_id.get(f"fcst_high:{st}")
        if fc is not None:
            f = fc[(fc["obs_ts"] == target_ts) & (fc["vintage_ts"] <= r.decision_ts)]
            if not f.empty:
                last = f.sort_values("vintage_ts").iloc[-1]
                row["wx_margin_nws"] = _margin(last["value"], r.strike_type, r.floor_strike,
                                               r.cap_strike)
                used.append(last["vintage_ts"])
        row["wx_ts"] = max(used) if used else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


# Kalshi CPI year-over-year series -> FRED series it settles on (BLS headline 12-month
# change = CPI-U, not seasonally adjusted, rounded to one decimal; checked 2026-10-29).
ECON_SERIES = {"KXCPIYOY": "CPIAUCNS", "CPIYOY": "CPIAUCNS"}
# FRED vintages have day resolution (midnight UTC) but BLS releases at 08:30 ET, so a value
# is treated as public only one full day after its vintage date.
FRED_PUBLIC_LAG = pd.Timedelta(days=1)
_EVENT_MONTH = re.compile(r"-(\d{2})([A-Z]{3})$")
_TICKER_STRIKE = re.compile(r"-T(-?\d+(?:\.\d+)?)$")


def _econ_strike(r) -> tuple[str | None, float | None, float | None]:
    if isinstance(r.strike_type, str) and r.strike_type:   # pandas gives NaN, not None
        return (r.strike_type, None if pd.isna(r.floor_strike) else r.floor_strike,
                None if pd.isna(r.cap_strike) else r.cap_strike)
    m = _TICKER_STRIKE.search(r.ticker)   # legacy tickers: "-T8.0" = "above 8.0%"
    return ("greater", float(m.group(1)), None) if m else (None, None, None)


def _economics(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time CPI features for CPI year-over-year markets.

    econ_margin_last  latest *published* YoY (one decimal, as BLS reports it) vs the strike
    econ_margin_proj  naive projection for the target month: last YoY plus the average of the
                      last 3 monthly changes minus the change dropping out of the 12-month
                      window (a base effect that is already known)
    econ_yoy_mom      change between the last two published YoY values
    """
    ec = base[base["series_ticker"].isin(ECON_SERIES)]
    if ec.empty:
        return pd.DataFrame({"ticker": base["ticker"]})
    with engine.connect() as c:
        obs = pd.DataFrame(c.execute(
            select(ext_observations.c.series_id, ext_observations.c.obs_ts,
                   ext_observations.c.vintage_ts, ext_observations.c.value)
            .where(ext_observations.c.source == "fred",
                   ext_observations.c.series_id.in_(set(ECON_SERIES.values())))).mappings().all())
    if obs.empty:
        return pd.DataFrame({"ticker": base["ticker"]})
    obs["obs_ts"], obs["vintage_ts"] = _to_ts(obs["obs_ts"]), _to_ts(obs["vintage_ts"])
    obs["public_ts"] = obs["vintage_ts"] + FRED_PUBLIC_LAG
    rows = []
    for r in ec.itertuples():
        row = {"ticker": r.ticker, "econ_margin_last": np.nan, "econ_margin_proj": np.nan,
               "econ_yoy_mom": np.nan, "econ_ts": pd.NaT}
        m = _EVENT_MONTH.search(r.event_ticker or "")
        st, floor, cap = _econ_strike(r)
        if not m or m.group(2) not in _MONTHS or st is None:
            rows.append(row)
            continue
        target = pd.Timestamp(year=2000 + int(m.group(1)), month=_MONTHS[m.group(2)], day=1,
                              tz="UTC")
        s = obs[(obs["series_id"] == ECON_SERIES[r.series_ticker])
                & (obs["public_ts"] <= r.decision_ts)].dropna(subset=["value"])
        # latest vintage of each month known at decision time
        s = s.sort_values("vintage_ts").groupby("obs_ts").tail(1).set_index("obs_ts")["value"]
        s = s[s.index < target].sort_index()
        if len(s) < 16:
            rows.append(row)
            continue
        idx = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS", tz="UTC"))
        yoy = (idx / idx.shift(12) - 1) * 100
        mom = np.log(idx).diff() * 100
        last = yoy.dropna()
        if len(last) < 2:
            rows.append(row)
            continue
        last_month = last.index[-1]
        yoy_last = round(float(last.iloc[-1]), 1)
        out_month = target - pd.DateOffset(months=12)
        drop_out = mom.get(out_month, np.nan)
        recent = mom.dropna().iloc[-3:].mean()
        gap = (target.year - last_month.year) * 12 + target.month - last_month.month
        # only defined one month ahead: YoY(t) ~ YoY(t-1) + mom(t) - mom(t-12)
        proj = yoy_last + recent - drop_out if gap == 1 else np.nan
        row["econ_margin_last"] = _margin(yoy_last, st, floor, cap)
        row["econ_margin_proj"] = _margin(proj, st, floor, cap) if np.isfinite(proj) else np.nan
        row["econ_yoy_mom"] = float(last.iloc[-1] - last.iloc[-2])
        row["econ_ts"] = obs[(obs["public_ts"] <= r.decision_ts)]["public_ts"].max()
        rows.append(row)
    return pd.DataFrame(rows)


FEATURE_SETS = {"market": _market_state, "per_market_obs": _per_market_obs, "weather": _weather,
                "economics": _economics}


def build_dataset(engine: Engine, *, source=None, category=None, series=None,
                  horizon: timedelta = timedelta(hours=6),
                  feature_sets=("market", "per_market_obs", "weather", "economics"),
                  use_market_features: bool = True, since=None, as_of=None) -> Dataset:
    base = load_markets(engine, source=source, category=category, series=series, as_of=as_of)
    if since is not None and not base.empty:
        base = base[base["close_time"] >= pd.Timestamp(since)].reset_index(drop=True)
    if base.empty:
        raise ValueError("no settled binary markets match the filters")
    base["decision_ts"] = base["close_time"] - horizon
    early = base["open_time"].notna() & (base["decision_ts"] < base["open_time"])
    base.loc[early, "decision_ts"] = base.loc[early, "open_time"]
    base["_early"] = early   # carried as a column so it survives the sort below
    base["label_ts"] = base["expiration_time"].fillna(base["close_time"])
    if as_of is not None:
        base["label_ts"] = base["label_ts"].clip(upper=pd.Timestamp(as_of))
    base = base.sort_values(["decision_ts", "ticker"]).reset_index(drop=True)
    y = (base["result"] == "yes").to_numpy(dtype=float)
    return _assemble(engine, base, base.pop("_early"), feature_sets, use_market_features, y)


def load_open_markets(engine: Engine, *, series=None, as_of) -> pd.DataFrame:
    """Binary markets open at ``as_of`` (open_time <= as_of < close_time), whatever their
    stored status says, so the same query serves live use and historical replay."""
    ts = pd.Timestamp(as_of).to_pydatetime()
    q = (select(mkt_markets.c.ticker, mkt_markets.c.source, mkt_markets.c.category,
                mkt_markets.c.series_ticker, mkt_markets.c.event_ticker, mkt_markets.c.open_time,
                mkt_markets.c.close_time, mkt_markets.c.expiration_time,
                mkt_markets.c.strike_type, mkt_markets.c.floor_strike, mkt_markets.c.cap_strike)
         .where(mkt_markets.c.open_time <= ts, mkt_markets.c.close_time > ts))
    if series is not None:
        q = q.where(mkt_markets.c.series_ticker.in_(series) if isinstance(
            series, (list, tuple, set)) else mkt_markets.c.series_ticker == series)
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(q).mappings().all())
    if df.empty:
        return df
    for col in ("open_time", "close_time", "expiration_time"):
        df[col] = _to_ts(df[col])
    return df


def build_live_dataset(engine: Engine, *, series, as_of, feature_names: list[str],
                       min_to_close: timedelta, max_to_close: timedelta,
                       feature_sets=("market", "per_market_obs", "weather", "economics")
                       ) -> Dataset:
    """Features for open markets whose time to close is in [min_to_close, max_to_close],
    with decision time = ``as_of``. Columns are aligned to a trained model's
    ``feature_names``; ``y`` is NaN because the outcome is unknown."""
    base = load_open_markets(engine, series=series, as_of=as_of)
    now = pd.Timestamp(as_of)
    if not base.empty:
        ttc = base["close_time"] - now
        base = base[(ttc >= min_to_close) & (ttc <= max_to_close)].reset_index(drop=True)
    if base.empty:
        return Dataset(pd.DataFrame(columns=feature_names), np.array([]), pd.DataFrame(
            columns=["ticker", "decision_ts", "market_mid", "dq_flags"]), list(feature_names))
    base["decision_ts"] = now
    base["label_ts"] = base["expiration_time"].fillna(base["close_time"])
    base["revised"] = False
    base = base.sort_values("ticker").reset_index(drop=True)
    ds = _assemble(engine, base, pd.Series(False, index=base.index), feature_sets, True,
                   np.full(len(base), np.nan))
    ds.X = ds.X.reindex(columns=feature_names)
    ds.feature_names = list(feature_names)
    return ds


def _assemble(engine: Engine, base: pd.DataFrame, early: pd.Series, feature_sets,
              use_market_features: bool, y: np.ndarray) -> Dataset:
    feats = base[["ticker"]].copy()
    ts_cols = []
    for name in feature_sets:
        part = FEATURE_SETS[name](engine, base)
        feats = feats.merge(part, on="ticker", how="left")
        ts_cols += [c for c in part.columns if c in ("mkt_ts", "obs_ts", "wx_ts", "econ_ts")]
    feats = base[["ticker"]].merge(feats, on="ticker", how="left")  # restore order

    input_max = feats[ts_cols].max(axis=1) if ts_cols else pd.Series(pd.NaT, index=feats.index)
    input_max = pd.to_datetime(input_max, utc=True)
    leaked = input_max.notna() & (input_max > base["decision_ts"])
    if leaked.any():
        raise LookAheadError(f"{int(leaked.sum())} rows use inputs after their decision time, "
                             f"e.g. {base.loc[leaked, 'ticker'].iloc[0]}")

    feature_cols = [c for c in feats.columns if c not in ("ticker", *ts_cols)]
    if not use_market_features:
        feature_cols = [c for c in feature_cols if c not in MARKET_FEATURES]
    feature_cols = [c for c in feature_cols if feats[c].notna().any()]

    flags = []
    for i in range(len(base)):
        f = []
        mid = feats.at[i, "mkt_mid"] if "mkt_mid" in feats else np.nan
        if pd.isna(mid):
            f.append("no_market_price")
        else:
            if feats.at[i, "mkt_stale_hours"] > STALE_HOURS:
                f.append("stale_market_price")
            if feats.at[i, "mkt_spread"] > WIDE_SPREAD:
                f.append("wide_spread")
        if feature_cols and feats.loc[i, feature_cols].isna().any():
            f.append("missing_features")
        if base.at[i, "revised"]:
            f.append("outcome_revised")
        if early[i]:
            f.append("decision_clamped_to_open")
        flags.append(f)

    meta = base[["ticker", "source", "category", "series_ticker", "event_ticker",
                 "decision_ts", "label_ts"]].copy()
    meta["input_max_ts"] = input_max.fillna(meta["decision_ts"])
    meta["market_mid"] = feats.get("mkt_mid", np.nan)
    meta["cluster"] = meta["event_ticker"].fillna(meta["ticker"])
    meta["dq_flags"] = flags
    X = feats[feature_cols].astype(float).reset_index(drop=True)
    return Dataset(X, y, meta, feature_cols)
