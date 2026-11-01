"""Kalshi account dashboard: trades, realized gains/losses, bot decisions, paper record.

Run:  .venv\\Scripts\\streamlit run executor\\dashboard.py
Reads only local files (executor/data/ledger.db, the live audit log, data/sim.db). The
"Sync now" button pulls fresh fills/settlements/balance from Kalshi (read-only calls).
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from executor.config import ExecutorSettings
from executor.ledger import load

ROOT = Path(__file__).resolve().parents[1]
TZ = "America/Detroit"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (bot, other)
GOOD, CRITICAL = "#0ca30c", "#d03b3b"        # status: gain / loss (always with a sign)
GRID = "rgba(128,128,128,0.15)"

st.set_page_config(page_title="Kalshi bot", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 3.5rem; max-width: 1200px;}
  [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-size: 1.7rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.8rem; text-transform: uppercase;
                                   letter-spacing: 0.04em; opacity: 0.7;}
  .badge {display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;}
  .live {background: rgba(208,59,59,0.12); color: #d03b3b;}
  .dry {background: rgba(12,163,12,0.12); color: #0a8a0a;}
  .warn {background: rgba(250,178,25,0.16); color: #9a6a00;}
  .stButton button {transition: transform 160ms ease-out;}
  .stButton button:active {transform: scale(0.97);}
</style>""", unsafe_allow_html=True)

cfg = ExecutorSettings()
DATA = cfg.data_dir


def money(x, signed=False):
    if x is None or pd.isna(x):
        return "—"
    s = f"${abs(x):,.2f}"
    if signed:
        return ("+" if x > 0 else "−" if x < 0 else "±") + s
    return ("−" if x < 0 else "") + s


def local(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(TZ)


def style_fig(fig: go.Figure, height=300) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 8, "r": 8, "t": 8, "b": 8},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="x unified", legend={"orientation": "h", "y": 1.08, "x": 0},
                      font={"size": 12})
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="rgba(128,128,128,0.45)",
                     tickformat="$,.2f", rangemode="tozero")
    return fig


def md(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't render dollar amounts as math."""
    return text.replace("$", "\\$")


def audit_events() -> pd.DataFrame:
    path = Path(DATA) / "audit_live.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["ts", "event"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(rows)
    df["ts"] = local(df["ts"])
    return df


# ---------------------------------------------------------------- header
left, right = st.columns([3, 1])
with left:
    mode = cfg.mode
    badge = ('<span class="badge live">LIVE · REAL MONEY</span>' if mode == "live" else
             f'<span class="badge dry">{mode.upper()}</span>')
    if cfg.gate_overridden:
        badge += ' <span class="badge warn">GATE OVERRIDDEN · NO VALIDATED EDGE</span>'
    st.markdown(f"## Kalshi bot &nbsp; {badge}", unsafe_allow_html=True)
with right:
    if st.button("Sync now", width="stretch", disabled=mode == "dry_run"):
        from executor.cli import _client
        from executor.ledger import sync
        c = _client(cfg)
        try:
            with st.spinner("Reading fills, settlements and balance from Kalshi…"):
                sync(c, DATA)
        except Exception as e:  # noqa: BLE001 - show it, don't crash the page
            st.error(f"Sync failed: {e}")
        finally:
            c.close()

fills, settles, bal = load(DATA)
events = audit_events()
if len(bal):
    st.caption(md(f"Last synced {local(bal['ts']).iloc[-1]:%b %d, %I:%M %p} · syncs hourly with "
               f"the trading job · limits: bankroll ${cfg.bankroll_cap_usd:g}, max order "
               f"${cfg.max_order_usd:g}, daily loss stop ${cfg.daily_loss_stop_usd:g}"))
else:
    st.info("No ledger yet. Click **Sync now** (or wait for the hourly job).")

# ---------------------------------------------------------------- KPIs
bot_s = settles[settles["source"] == "bot"]
oth_s = settles[settles["source"] == "other"]
bot_f = fills[fills["source"] == "bot"]
equity = bal["equity"].iloc[-1] if len(bal) else None
cash = bal["cash"].iloc[-1] if len(bal) else None
open_val = bal["positions_value"].iloc[-1] if len(bal) else None
wins = int((bot_s["realized_pnl"] > 0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Account value", money(equity), help="Cash + current value of open positions")
k2.metric("Bot realized P&L", money(bot_s["realized_pnl"].sum(), signed=True),
          help="Settled markets the bot traded: payout − cost − fees")
k3.metric("Open positions", money(open_val), help=md(f"Cash available: {money(cash)}"))
k4.metric("Bot trades", f"{len(bot_f)}",
          help=f"{len(bot_s)} settled · {wins} won · {len(bot_s) - wins} lost")

tab_pnl, tab_trades, tab_bot, tab_paper = st.tabs(
    ["Gains & losses", "All trades", "Bot decisions", "Paper record"])

# ---------------------------------------------------------------- gains & losses
with tab_pnl:
    st.markdown("##### Cumulative realized P&L")
    if settles.empty:
        st.caption("Nothing has settled yet.")
    else:
        fig = go.Figure()
        for src, color, name in (("bot", BLUE, "Bot"), ("other", ORANGE, "Other activity")):
            d = settles[settles["source"] == src]
            if d.empty:
                continue
            fig.add_scatter(x=local(d["settled_ts"]), y=d["realized_pnl"].cumsum(), name=name,
                            mode="lines+markers", line={"color": color, "width": 2, "shape": "hv"},
                            marker={"size": 8, "line": {"width": 2, "color": "white"}},
                            hovertemplate="%{y:$,.2f}<extra>" + name + "</extra>")
        fig.update_layout(showlegend=True)
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown("##### Each settled market")
        d = settles.sort_values("settled_ts")
        fig = go.Figure(go.Bar(
            x=local(d["settled_ts"]), y=d["realized_pnl"],
            marker={"color": [GOOD if v > 0 else CRITICAL for v in d["realized_pnl"]],
                        "cornerradius": 4},
            customdata=d[["ticker", "result", "source"]],
            hovertemplate="%{customdata[0]}<br>result %{customdata[1]} · %{customdata[2]}"
                          "<br>%{y:+$,.2f}<extra></extra>"))
        st.plotly_chart(style_fig(fig, 240), width="stretch")

    st.markdown("##### Account value over time")
    if len(bal) >= 2:
        fig = go.Figure(go.Scatter(x=local(bal["ts"]), y=bal["equity"], mode="lines",
                                   line={"color": BLUE, "width": 2}, name="Account value",
                                   hovertemplate="%{y:$,.2f}<extra></extra>"))
        st.plotly_chart(style_fig(fig, 240), width="stretch")
        st.caption("Deposits and withdrawals show up as jumps here; realized P&L above "
                   "excludes them.")
    else:
        st.caption("Builds up from hourly balance snapshots.")

# ---------------------------------------------------------------- all trades
with tab_trades:
    show = st.segmented_control("Source", ["All", "Bot", "Other"], default="All")
    f = fills if show in (None, "All") else fills[fills["source"] == show.lower()]
    st.markdown(f"##### Fills ({len(f)})")
    if f.empty:
        st.caption("No fills.")
    else:
        t = f.sort_values("ts", ascending=False)
        st.dataframe(pd.DataFrame({
            "Time": local(t["ts"]).dt.strftime("%b %d %I:%M %p"), "Market": t["ticker"],
            "Bought": t["outcome"].str.upper(), "Action": t["action"],
            "Contracts": t["count"], "Price": t["price"], "Cost": t["count"] * t["price"],
            "Fee": t["fee"], "Source": t["source"]}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="$%.2f")
                           for c in ("Price", "Cost", "Fee")})
    s = settles if show in (None, "All") else settles[settles["source"] == show.lower()]
    st.markdown(f"##### Settlements ({len(s)})")
    if s.empty:
        st.caption("No settlements.")
    else:
        t = s.sort_values("settled_ts", ascending=False)
        st.dataframe(pd.DataFrame({
            "Settled": local(t["settled_ts"]).dt.strftime("%b %d %I:%M %p"),
            "Market": t["ticker"], "Result": t["result"].str.upper(),
            "Cost": t["cost"], "Payout": t["revenue"], "Fees": t["settle_fee"],
            "P&L": t["realized_pnl"],
            "": ["▲ gain" if v > 0 else "▼ loss" if v < 0 else "even" for v in t["realized_pnl"]],
            "Source": t["source"]}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="$%.2f")
                           for c in ("Cost", "Payout", "Fees", "P&L")})

# ---------------------------------------------------------------- bot decisions
with tab_bot:
    if events.empty:
        st.caption("No live executor activity yet.")
    else:
        kinds = {"order_submit": "Orders sent", "order_ack": "Exchange acks",
                 "order_blocked": "Blocked by limits", "order_error": "Errors",
                 "cycle_start": "Cycles run"}
        cols = st.columns(len(kinds))
        for col, (k, label) in zip(cols, kinds.items()):
            col.metric(label, int((events["event"] == k).sum()))
        st.markdown("##### Recent activity")
        ev = events.sort_values("ts", ascending=False).head(200)
        detail = []
        for r in ev.itertuples():
            plan = getattr(r, "plan", None)
            if isinstance(plan, dict):
                detail.append(f"{plan.get('outcome', '').upper()} {plan.get('count')}× "
                              f"{plan.get('ticker')} ≤ ${plan.get('max_price')}")
            elif isinstance(getattr(r, "reason", None), str):
                detail.append(r.reason)
            elif isinstance(getattr(r, "ticker", None), str):
                detail.append(r.ticker)
            else:
                detail.append("")
        st.dataframe(pd.DataFrame({"Time": ev["ts"].dt.strftime("%b %d %I:%M:%S %p"),
                                   "Event": ev["event"], "Detail": detail}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- paper record
with tab_paper:
    st.caption("PAPER TRADING — NOT REAL MONEY. The research loop's simulated record for "
               "the same forecasts; this is the evidence the go-live gate reads.")
    from sqlalchemy import create_engine, select

    from predlab.db.schema import model_registry, sim_equity
    sim = create_engine(f"sqlite:///{ROOT / 'data' / 'sim.db'}")
    with sim.connect() as c:
        regs = c.execute(select(model_registry)).mappings().all()
    for r in regs:
        g = (r["validation_json"] or {}).get("gate", {})
        st.markdown(f"**{r['model_version']}** · status `{r['status']}`")
        a, b, c3, d = st.columns(4)
        a.metric("Paper days", f"{g.get('paper_days', 0)} / 28")
        b.metric("Forecasts", g.get("paper_forecasts", 0),
                 help=f"{g.get('paper_resolved', 0)} resolved")
        c3.metric("Paper P&L", money(g.get("paper_net_pnl"), signed=True))
        diff = g.get("paper_brier_diff_vs_market")
        d.metric("Vs market (Brier)", "—" if diff is None else f"{diff:+.4f}",
                 help="Below 0 = more accurate than the market price")
        with sim.connect() as c:
            eq = pd.DataFrame(c.execute(select(sim_equity).where(
                sim_equity.c.run_id == f"paper-{r['model_version']}")).mappings().all())
        if len(eq) >= 2:
            fig = go.Figure(go.Scatter(x=local(eq["ts"]), y=eq["net_pnl"], mode="lines",
                                       line={"color": BLUE, "width": 2}, name="Paper net P&L",
                                       hovertemplate="%{y:+$,.2f}<extra></extra>"))
            st.plotly_chart(style_fig(fig, 220), width="stretch")
    if not regs:
        st.caption("No deployments registered.")
