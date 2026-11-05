"""Kalshi Desktop Control Widget:
High-speed native GUI with instant open/close (no subprocess lag, no empty console shells).

Controls & Features:
1. Trading Bot (Windows Task Scheduler loop: Enable / Disable via native schtasks.exe)
2. Clamshell Mode (Keep Awake when laptop lid closed vs Normal Sleep via winreg & powercfg)
3. One-click Run Cycle Now
4. One-click Open Dashboard
5. Real-time balance and next-run status
6. System Health, Trade Blockers, and Diagnostics
7. Recent Meetings & Execution Feed
"""

import os
import sys
import subprocess
import sqlite3
import threading
import json
import time
import winreg
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MAIN_PROJECT = Path("C:/Users/brend/orca/projects/Personal Projects")


def _resolve(rel: str) -> Path:
    p = ROOT / rel
    if not p.exists() and (MAIN_PROJECT / rel).exists():
        return MAIN_PROJECT / rel
    return p


PY = _resolve(".venv/Scripts/python.exe")
PYW = _resolve(".venv/Scripts/pythonw.exe")
LEDGER_DB = _resolve("executor/data/ledger.db")
SIM_DB = _resolve("data/sim.db")
RUN_LOOP_PY = _resolve("scripts/run_loop.py")
DASHBOARD_CMD = _resolve("scripts/dashboard.cmd")
AUDIT_FILE = _resolve("executor/data/audit_live.jsonl")
SEARCH_USAGE = _resolve("data/desk_search_usage.json")

# Flag to hide Windows console on spawned subprocesses
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def get_bot_status() -> tuple[bool, str, str]:
    """Fast native Task Scheduler check via schtasks.exe (< 80ms)."""
    try:
        res = subprocess.run(
            ["schtasks.exe", "/query", "/tn", "predlab paper loop", "/fo", "csv", "/nh"],
            capture_output=True, text=True, creationflags=NO_WINDOW, timeout=2.5
        )
        if res.returncode != 0 or not res.stdout.strip():
            return False, "Not Found", "N/A"
        # CSV output: "\predlab paper loop","11/4/2026 7:35:00 PM","Ready"
        line = res.stdout.strip()
        parts = [p.strip('"\r\n ') for p in line.split('","')]
        state = parts[2] if len(parts) > 2 else "Unknown"
        raw_next = parts[1] if len(parts) > 1 else "N/A"
        try:
            dt = datetime.strptime(raw_next, "%m/%d/%Y %I:%M:%S %p")
            next_run = dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            next_run = raw_next
        is_enabled = (state.lower() not in ("disabled", "not found", "unknown"))
        return is_enabled, state, next_run
    except Exception as e:
        return False, str(e), "N/A"


def set_bot_enabled(enable: bool) -> bool:
    action = "/enable" if enable else "/disable"
    try:
        subprocess.run(
            ["schtasks.exe", "/change", "/tn", "predlab paper loop", action],
            capture_output=True, text=True, creationflags=NO_WINDOW, timeout=4.0
        )
        enabled, _, _ = get_bot_status()
        return enabled == enable
    except Exception:
        return False


def get_lid_status() -> bool:
    """Sub-millisecond lid setting lookup via Python's native winreg."""
    try:
        key = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes\381b4222-f694-41f0-9685-ff5bb260df2e\4f971e89-eebd-4455-a8de-9e59040e7347\5ca83367-6e45-459f-a27b-476b1d01c936"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            val, _ = winreg.QueryValueEx(k, "ACSettingIndex")
            return val == 0
    except Exception:
        return False


def set_lid_mode(awake: bool) -> None:
    """Direct fast powercfg invocation (no PowerShell overhead)."""
    val = "0" if awake else "1"
    standby = "0" if awake else "1800"
    try:
        subprocess.run(["powercfg.exe", "/setacvalueindex", "SCHEME_CURRENT",
                        "4f971e89-eebd-4455-a8de-9e59040e7347", "5ca83367-6e45-459f-a27b-476b1d01c936", val],
                       creationflags=NO_WINDOW, check=False)
        subprocess.run(["powercfg.exe", "/setdcvalueindex", "SCHEME_CURRENT",
                        "4f971e89-eebd-4455-a8de-9e59040e7347", "5ca83367-6e45-459f-a27b-476b1d01c936", val],
                       creationflags=NO_WINDOW, check=False)
        subprocess.run(["powercfg.exe", "/setacvalueindex", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE", standby],
                       creationflags=NO_WINDOW, check=False)
        subprocess.run(["powercfg.exe", "/setactive", "SCHEME_CURRENT"],
                       creationflags=NO_WINDOW, check=False)
    except Exception:
        pass


def get_account_info() -> tuple[float, float, float, float]:
    """Returns (total, cash, positions, today_pnl) from local ledger (< 5ms)."""
    total, cash, pos, today_pnl = 0.0, 0.0, 0.0, 0.0
    try:
        if LEDGER_DB.is_file():
            with sqlite3.connect(LEDGER_DB, timeout=2.0) as c:
                row = c.execute("SELECT cash, positions_value FROM balances ORDER BY rowid DESC LIMIT 1").fetchone()
                if row:
                    cash, pos = float(row[0]), float(row[1])
                    total = cash + pos
                first_row = c.execute(
                    "SELECT cash, positions_value FROM balances WHERE ts >= date('now', 'start of day') ORDER BY rowid ASC LIMIT 1"
                ).fetchone()
                if first_row:
                    today_pnl = total - (float(first_row[0]) + float(first_row[1]))
    except Exception:
        pass
    return total, cash, pos, today_pnl


def get_diagnostics() -> tuple[list[dict], list[dict], str]:
    """Audits recent meetings and cycles for trade blockers (< 10ms)."""
    issues = []
    recent_meetings = []
    last_fill_text = "No trades recorded."

    # 1. Meetings & dry streak analysis
    streak = 0
    if SIM_DB.is_file():
        try:
            with sqlite3.connect(SIM_DB, timeout=2.0) as c:
                c.row_factory = sqlite3.Row
                rows = c.execute(
                    "SELECT id, ts, tickers_json, decisions_json, transcript "
                    "FROM desk_meetings ORDER BY id DESC LIMIT 15"
                ).fetchall()
                counting = True
                for r in rows:
                    decs = json.loads(r["decisions_json"] or "[]")
                    buys = [d for d in decs if d.get("action") == "BUY"]
                    tickers = json.loads(r["tickers_json"] or "[]")
                    if counting:
                        if not buys:
                            streak += 1
                        else:
                            counting = False

                    trans = r["transcript"] or ""
                    if buys:
                        summary = f"{len(buys)} BUY(s) executed"
                    elif "NOTHING UNUSUAL" in trans:
                        summary = "Scanner: Nothing unusual"
                    elif "daily search cap" in trans:
                        summary = "Search cap reached (no clear info)"
                    else:
                        summary = "No edge > 5c after fees"

                    ts_val = r["ts"]
                    ts_short = ts_val[11:16] if len(ts_val) >= 16 else ts_val
                    recent_meetings.append({
                        "id": r["id"],
                        "ts": ts_short,
                        "scanned": len(tickers),
                        "buys": len(buys),
                        "summary": summary
                    })
        except Exception as e:
            issues.append({"level": "ERROR", "title": f"sim.db read error: {e}", "fix": "Check database file permissions."})

    if streak >= 4:
        issues.append({
            "level": "WARNING",
            "title": f"Dry Streak: Last {streak} meetings had 0 purchases",
            "fix": "Normal during quiet hours. Check spot feeds or trigger 'Run Meeting Now' above."
        })

    # 2. Audit log checks
    if AUDIT_FILE.is_file():
        try:
            lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
            for l in reversed(lines[-35:]):
                rec = json.loads(l)
                if rec.get("event") == "order_blocked":
                    reason = rec.get("reason", "")
                    ticker = rec.get("ticker", "")
                    if "bankroll cap" in reason:
                        issues.append({
                            "level": "NOTICE",
                            "title": f"Exposure Cap: $40 bankroll limit reached ({ticker})",
                            "fix": "Waiting for open positions to settle; or raise cap in executor/.env."
                        })
                    elif "insufficient balance" in reason:
                        issues.append({
                            "level": "NOTICE",
                            "title": f"Cash Limit: Kalshi subaccount cash low for {ticker}",
                            "fix": "Kalshi isolates funds by exchange; balance frees up on settlement."
                        })
                    break
        except Exception:
            pass

    # 3. Web Search quota usage
    if SEARCH_USAGE.is_file():
        try:
            u = json.loads(SEARCH_USAGE.read_text(encoding="utf-8"))
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            used = u.get(today_str, 0)
            if used >= 60:
                issues.append({
                    "level": "NOTICE",
                    "title": f"Web Search Cap: {used}/60 daily queries reached",
                    "fix": "Live spot feeds continue pricing crypto/commodities/weather; web search resets 00:00 UTC."
                })
        except Exception:
            pass

    # 4. Check available cash
    total, cash, pos, _ = get_account_info()
    if cash < 3.0 and total > 0:
        issues.append({
            "level": "WARNING",
            "title": f"Low Cash: ${cash:.2f} available (< $3.00 min order)",
            "fix": "Wait for open positions to resolve or deposit cash on Kalshi."
        })

    # 5. Last executed trade
    if LEDGER_DB.is_file():
        try:
            with sqlite3.connect(LEDGER_DB, timeout=2.0) as c:
                c.row_factory = sqlite3.Row
                f = c.execute("SELECT ts, ticker, action, outcome, count, price FROM fills ORDER BY ts DESC LIMIT 1").fetchone()
                if f:
                    ts_str = f["ts"][:16]
                    cnt = f["count"]
                    cnt_str = f"{cnt:.0f}" if cnt == int(cnt) else f"{cnt:.2f}"
                    last_fill_text = f"{f['action'].upper()} {cnt_str}x {f['ticker']} ({f['outcome'].upper()}) @ ${f['price']:.2f}  [{ts_str}]"
        except Exception:
            pass

    return issues, recent_meetings, last_fill_text


class KalshiControlApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kalshi Control Panel")
        self.root.geometry("460x780")
        self.root.resizable(False, False)
        self.root.configure(bg="#121316")

        self.running_cycle = False
        self._refresh_timer = None

        # Clean, instant termination on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_status()
        self._schedule_periodic_refresh()

    def _on_close(self):
        """Immediately destroys the window and exits the process with 0 delay."""
        if self._refresh_timer is not None:
            try:
                self.root.after_cancel(self._refresh_timer)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    def _build_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#121316")
        header_frame.pack(fill="x", padx=16, pady=(14, 8))

        title = tk.Label(header_frame, text="KALSHI DESK CONTROL", font=("Segoe UI", 13, "bold"),
                         fg="#f0f2f5", bg="#121316")
        title.pack(side="left")

        self.live_badge = tk.Label(header_frame, text="● LIVE", font=("Segoe UI", 9, "bold"),
                                   fg="#0ca30c", bg="#121316")
        self.live_badge.pack(side="right")

        # Card 1: Account Metrics
        card_acc = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_acc.pack(fill="x", padx=16, pady=4)

        acc_inner = tk.Frame(card_acc, bg="#1c1e24")
        acc_inner.pack(fill="x", padx=14, pady=9)

        self.lbl_equity = tk.Label(acc_inner, text="$0.00", font=("Segoe UI", 20, "bold"),
                                   fg="#f0f2f5", bg="#1c1e24")
        self.lbl_equity.pack(anchor="w")

        self.lbl_split = tk.Label(acc_inner, text="Cash: $0.00  |  Positions: $0.00",
                                  font=("Segoe UI", 9), fg="#9ba1a6", bg="#1c1e24")
        self.lbl_split.pack(anchor="w", pady=(2, 2))

        self.lbl_pnl = tk.Label(acc_inner, text="Today: +$0.00  (Goal: $15.00)",
                                font=("Segoe UI", 10, "bold"), fg="#0ca30c", bg="#1c1e24")
        self.lbl_pnl.pack(anchor="w")

        # Card 2: Controls (Bot & Lid Toggles)
        card_toggles = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_toggles.pack(fill="x", padx=16, pady=4)

        toggles_inner = tk.Frame(card_toggles, bg="#1c1e24")
        toggles_inner.pack(fill="x", padx=14, pady=9)

        # Bot Toggle Row
        bot_row = tk.Frame(toggles_inner, bg="#1c1e24")
        bot_row.pack(fill="x", pady=(0, 6))

        bot_left = tk.Frame(bot_row, bg="#1c1e24")
        bot_left.pack(side="left", fill="both", expand=True)

        tk.Label(bot_left, text="Trading Bot (15m Loop)", font=("Segoe UI", 9, "bold"),
                 fg="#f0f2f5", bg="#1c1e24").pack(anchor="w")
        self.lbl_bot_desc = tk.Label(bot_left, text="Checking status...", font=("Segoe UI", 8),
                                     fg="#9ba1a6", bg="#1c1e24")
        self.lbl_bot_desc.pack(anchor="w")

        self.btn_bot_toggle = tk.Button(bot_row, text="--", font=("Segoe UI", 9, "bold"),
                                        width=7, relief="flat", cursor="hand2", command=self._toggle_bot)
        self.btn_bot_toggle.pack(side="right")

        # Separator line
        sep = tk.Frame(toggles_inner, bg="#2a2d36", height=1)
        sep.pack(fill="x", pady=4)

        # Lid Toggle Row
        lid_row = tk.Frame(toggles_inner, bg="#1c1e24")
        lid_row.pack(fill="x", pady=(4, 0))

        lid_left = tk.Frame(lid_row, bg="#1c1e24")
        lid_left.pack(side="left", fill="both", expand=True)

        tk.Label(lid_left, text="Run When Laptop Closed", font=("Segoe UI", 9, "bold"),
                 fg="#f0f2f5", bg="#1c1e24").pack(anchor="w")
        self.lbl_lid_desc = tk.Label(lid_left, text="Checking power plan...", font=("Segoe UI", 8),
                                     fg="#9ba1a6", bg="#1c1e24")
        self.lbl_lid_desc.pack(anchor="w")

        self.btn_lid_toggle = tk.Button(lid_row, text="--", font=("Segoe UI", 9, "bold"),
                                        width=7, relief="flat", cursor="hand2", command=self._toggle_lid)
        self.btn_lid_toggle.pack(side="right")

        # Action Buttons
        btn_frame = tk.Frame(self.root, bg="#121316")
        btn_frame.pack(fill="x", padx=16, pady=4)

        self.btn_run_now = tk.Button(btn_frame, text="⚡ Run Meeting & Cycle Now", font=("Segoe UI", 9, "bold"),
                                     bg="#2a78d6", fg="#ffffff", activebackground="#1f5bb5", activeforeground="#ffffff",
                                     relief="flat", cursor="hand2", height=1, command=self._run_cycle_now)
        self.btn_run_now.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_dashboard = tk.Button(btn_frame, text="📈 Live Dashboard", font=("Segoe UI", 9, "bold"),
                                       bg="#272a34", fg="#f0f2f5", activebackground="#343845", activeforeground="#ffffff",
                                       relief="flat", cursor="hand2", height=1, command=self._open_dashboard)
        self.btn_dashboard.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Card 3: Trade Blockers & Diagnostics
        self.card_diag = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        self.card_diag.pack(fill="x", padx=16, pady=5)

        diag_inner = tk.Frame(self.card_diag, bg="#1c1e24")
        diag_inner.pack(fill="x", padx=14, pady=9)

        diag_hdr = tk.Frame(diag_inner, bg="#1c1e24")
        diag_hdr.pack(fill="x", pady=(0, 6))

        tk.Label(diag_hdr, text="SYSTEM HEALTH & TRADE BLOCKERS", font=("Segoe UI", 9, "bold"),
                 fg="#cbd5e1", bg="#1c1e24").pack(side="left")

        self.lbl_health_badge = tk.Label(diag_hdr, text="● ALL CLEAR", font=("Segoe UI", 8, "bold"),
                                         fg="#0ca30c", bg="#1c1e24")
        self.lbl_health_badge.pack(side="right")

        self.diag_content_frame = tk.Frame(diag_inner, bg="#1c1e24")
        self.diag_content_frame.pack(fill="x")

        # Card 4: Recent Meetings & Trading Feed
        card_feed = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_feed.pack(fill="both", expand=True, padx=16, pady=4)

        feed_inner = tk.Frame(card_feed, bg="#1c1e24")
        feed_inner.pack(fill="both", expand=True, padx=14, pady=9)

        feed_hdr = tk.Frame(feed_inner, bg="#1c1e24")
        feed_hdr.pack(fill="x", pady=(0, 4))

        tk.Label(feed_hdr, text="RECENT MEETINGS & ACTIVITY", font=("Segoe UI", 9, "bold"),
                 fg="#cbd5e1", bg="#1c1e24").pack(side="left")

        self.lbl_last_trade = tk.Label(feed_inner, text="Last Trade: Loading...", font=("Segoe UI", 8),
                                       fg="#93c5fd", bg="#1c1e24")
        self.lbl_last_trade.pack(anchor="w", pady=(0, 6))

        self.meeting_rows_frame = tk.Frame(feed_inner, bg="#1c1e24")
        self.meeting_rows_frame.pack(fill="both", expand=True)

        # Footer
        self.lbl_footer = tk.Label(self.root, text="Ready", font=("Segoe UI", 8), fg="#6b7280", bg="#121316")
        self.lbl_footer.pack(side="bottom", pady=6)

    def _refresh_status(self):
        # 1. Fast Account info
        total, cash, pos, today_pnl = get_account_info()
        self.lbl_equity.config(text=f"${total:.2f}")
        self.lbl_split.config(text=f"Cash: ${cash:.2f}  |  Positions: ${pos:.2f}")
        sign = "+" if today_pnl >= 0 else "-"
        pnl_color = "#0ca30c" if today_pnl >= 0 else "#e05656"
        self.lbl_pnl.config(text=f"Today: {sign}${abs(today_pnl):.2f}  (Goal: $15.00)", fg=pnl_color)

        # 2. Fast Bot status (< 80ms)
        bot_enabled, bot_state, next_run = get_bot_status()
        if bot_enabled:
            self.lbl_bot_desc.config(text=f"ACTIVE • Next meeting at {next_run}", fg="#0ca30c")
            self.btn_bot_toggle.config(text="ON", bg="#0ca30c", fg="#ffffff", activebackground="#098209")
        else:
            self.lbl_bot_desc.config(text="PAUSED • Task scheduler disabled", fg="#e05656")
            self.btn_bot_toggle.config(text="OFF", bg="#3a3f4d", fg="#9ba1a6", activebackground="#2c303b")

        # 3. Fast Lid status (< 1ms via winreg)
        lid_awake = get_lid_status()
        if lid_awake:
            self.lbl_lid_desc.config(text="KEEP AWAKE • Trades with lid closed", fg="#0ca30c")
            self.btn_lid_toggle.config(text="ON", bg="#0ca30c", fg="#ffffff", activebackground="#098209")
        else:
            self.lbl_lid_desc.config(text="NORMAL SLEEP • Sleeps when lid closed", fg="#9ba1a6")
            self.btn_lid_toggle.config(text="OFF", bg="#3a3f4d", fg="#9ba1a6", activebackground="#2c303b")

        # 4. Fast Diagnostics & Issues (< 10ms)
        issues, meetings, last_fill = get_diagnostics()

        if not bot_enabled:
            issues.insert(0, {
                "level": "WARNING",
                "title": "Trading Bot is PAUSED (Loop scheduler disabled)",
                "fix": "Toggle the switch above to ON to resume automated meetings."
            })

        if any(i["level"] == "ERROR" for i in issues):
            self.lbl_health_badge.config(text=f"● {len(issues)} ERROR(S)", fg="#ef4444")
        elif any(i["level"] == "WARNING" for i in issues):
            self.lbl_health_badge.config(text=f"● {len(issues)} WARNING(S)", fg="#f59e0b")
        elif issues:
            self.lbl_health_badge.config(text=f"● {len(issues)} NOTICE(S)", fg="#38bdf8")
        else:
            self.lbl_health_badge.config(text="● ALL CLEAR", fg="#0ca30c")

        for widget in self.diag_content_frame.winfo_children():
            widget.destroy()

        if not issues:
            tk.Label(self.diag_content_frame, text="✅ No blockers detected. Loop and risk checks operating normally.",
                     font=("Segoe UI", 8), fg="#9ca3af", bg="#1c1e24").pack(anchor="w")
        else:
            for item in issues[:3]:
                box = tk.Frame(self.diag_content_frame, bg="#1c1e24")
                box.pack(fill="x", pady=2)

                color = "#ef4444" if item["level"] == "ERROR" else "#f59e0b" if item["level"] == "WARNING" else "#38bdf8"
                tk.Label(box, text=f"• {item['title']}", font=("Segoe UI", 8, "bold"),
                         fg=color, bg="#1c1e24", wraplength=410, justify="left").pack(anchor="w")
                tk.Label(box, text=f"   Fix: {item['fix']}", font=("Segoe UI", 7),
                         fg="#94a3b8", bg="#1c1e24", wraplength=410, justify="left").pack(anchor="w")

        # 5. Populate Feed
        self.lbl_last_trade.config(text=f"Last Trade: {last_fill}")

        for widget in self.meeting_rows_frame.winfo_children():
            widget.destroy()

        for m in meetings[:4]:
            row = tk.Frame(self.meeting_rows_frame, bg="#1c1e24")
            row.pack(fill="x", pady=2)

            left = tk.Frame(row, bg="#1c1e24")
            left.pack(side="left")

            badge_color = "#0ca30c" if m["buys"] > 0 else "#6b7280"
            tk.Label(left, text=f"M#{m['id']}", font=("Segoe UI", 8, "bold"), fg=badge_color, bg="#1c1e24").pack(side="left")
            tk.Label(left, text=f" ({m['ts']})", font=("Segoe UI", 8), fg="#9ca3af", bg="#1c1e24").pack(side="left")
            tk.Label(left, text=f" [{m['scanned']} mkts]", font=("Segoe UI", 8), fg="#64748b", bg="#1c1e24").pack(side="left")

            summary_color = "#86efac" if m["buys"] > 0 else "#94a3b8"
            tk.Label(row, text=m["summary"], font=("Segoe UI", 8), fg=summary_color, bg="#1c1e24", anchor="e").pack(side="right")

    def _schedule_periodic_refresh(self):
        self._refresh_timer = self.root.after(4000, self._auto_refresh)

    def _auto_refresh(self):
        if not self.running_cycle:
            self._refresh_status()
        self._schedule_periodic_refresh()

    def _toggle_bot(self):
        enabled, _, _ = get_bot_status()
        new_state = not enabled
        self.btn_bot_toggle.config(state="disabled", text="...")
        self.root.update()

        def worker():
            set_bot_enabled(new_state)
            self.root.after(0, self._on_bot_toggled)

        threading.Thread(target=worker, daemon=True).start()

    def _on_bot_toggled(self):
        self.btn_bot_toggle.config(state="normal")
        self._refresh_status()

    def _toggle_lid(self):
        awake = get_lid_status()
        new_mode = not awake
        self.btn_lid_toggle.config(state="disabled", text="...")
        self.root.update()

        def worker():
            set_lid_mode(new_mode)
            self.root.after(0, self._on_lid_toggled)

        threading.Thread(target=worker, daemon=True).start()

    def _on_lid_toggled(self):
        self.btn_lid_toggle.config(state="normal")
        self._refresh_status()

    def _run_cycle_now(self):
        if self.running_cycle:
            return
        self.running_cycle = True
        self.btn_run_now.config(state="disabled", text="⏳ Running...")
        self.lbl_footer.config(text="Executing desk meeting, loop and orders...", fg="#2a78d6")

        def worker():
            try:
                subprocess.run([str(PY), str(RUN_LOOP_PY)], cwd=ROOT, check=False,
                               creationflags=NO_WINDOW)
            finally:
                self.root.after(0, self._on_cycle_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _on_cycle_finished(self):
        self.running_cycle = False
        self.btn_run_now.config(state="normal", text="⚡ Run Meeting & Cycle Now")
        self.lbl_footer.config(text="Cycle completed", fg="#0ca30c")
        self._refresh_status()

    def _open_dashboard(self):
        try:
            subprocess.Popen([str(DASHBOARD_CMD)], cwd=ROOT, shell=True,
                             creationflags=NO_WINDOW)
            self.lbl_footer.config(text="Dashboard launched at http://localhost:8501", fg="#0ca30c")
        except Exception as e:
            self.lbl_footer.config(text=f"Dashboard error: {e}", fg="#e05656")


def main():
    root = tk.Tk()
    app = KalshiControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
