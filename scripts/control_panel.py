"""Kalshi Desktop Control Widget:
One-click toggles for:
1. Trading Bot (Task Scheduler loop: Enable / Disable)
2. Clamshell Mode (Keep Awake when laptop lid is closed vs Normal Sleep)
3. One-click Run Cycle Now
4. One-click Open Dashboard
5. Real-time balance and next-run status
"""

import os
import subprocess
import sqlite3
import threading
import time
import tkinter as tk
from pathlib import Path

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


def run_ps(cmd: str) -> str:
    res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return res.stdout.strip()


def get_bot_status() -> tuple[bool, str, str]:
    """Returns (is_enabled, state, next_run)"""
    try:
        cmd = (
            "$t = Get-ScheduledTask -TaskName 'predlab paper loop' -ErrorAction SilentlyContinue; "
            "if ($t) { "
            "  $info = Get-ScheduledTaskInfo -TaskName 'predlab paper loop' -ErrorAction SilentlyContinue; "
            "  $next = if ($info.NextRunTime) { $info.NextRunTime.ToString('h:mm tt') } else { 'N/A' }; "
            "  Write-Output \"$($t.State)|$next\" "
            "} else { Write-Output 'NotFound|N/A' }"
        )
        out = run_ps(cmd)
        parts = out.split("|")
        state = parts[0] if len(parts) > 0 else "Unknown"
        next_run = parts[1] if len(parts) > 1 else "N/A"
        is_enabled = (state != "Disabled" and state != "NotFound")
        return is_enabled, state, next_run
    except Exception as e:
        return False, str(e), "N/A"


def set_bot_enabled(enable: bool) -> bool:
    action = "Enable-ScheduledTask" if enable else "Disable-ScheduledTask"
    run_ps(f"{action} -TaskName 'predlab paper loop'")
    enabled, _, _ = get_bot_status()
    return enabled == enable


def get_lid_status() -> bool:
    """Returns True if lid is set to Do Nothing (0), False if Sleep (1)."""
    try:
        cmd = (
            "$r = Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes\\"
            "381b4222-f694-41f0-9685-ff5bb260df2e\\4f971e89-eebd-4455-a8de-9e59040e7347\\"
            "5ca83367-6e45-459f-a27b-476b1d01c936' -ErrorAction SilentlyContinue; "
            "if ($r -and $r.ACSettingIndex -eq 0) { 'Awake' } else { 'Sleep' }"
        )
        out = run_ps(cmd)
        return out.strip() == "Awake"
    except Exception:
        return False


def set_lid_mode(awake: bool) -> None:
    val = "0" if awake else "1"
    standby = "0" if awake else "1800"
    cmd = (
        f"powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 {val}; "
        f"powercfg /setdcvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 {val}; "
        f"powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE {standby}; "
        f"powercfg /setactive SCHEME_CURRENT"
    )
    run_ps(cmd)


def get_account_info() -> tuple[float, float, float, float]:
    """Returns (total, cash, positions, today_pnl)"""
    total, cash, pos, today_pnl = 0.0, 0.0, 0.0, 0.0
    try:
        if LEDGER_DB.is_file():
            with sqlite3.connect(LEDGER_DB) as c:
                row = c.execute("SELECT cash, positions_value FROM balances ORDER BY rowid DESC LIMIT 1").fetchone()
                if row:
                    cash, pos = float(row[0]), float(row[1])
                    total = cash + pos
                # Today's starting balance
                first_row = c.execute(
                    "SELECT cash, positions_value FROM balances WHERE ts >= date('now', 'start of day') ORDER BY rowid ASC LIMIT 1"
                ).fetchone()
                if first_row:
                    today_pnl = total - (float(first_row[0]) + float(first_row[1]))
    except Exception:
        pass
    return total, cash, pos, today_pnl


class KalshiControlApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kalshi Control Panel")
        self.root.geometry("420 x 510".replace(" ", ""))
        self.root.resizable(False, False)
        self.root.configure(bg="#121316")

        self.running_cycle = False
        self._build_ui()
        self._refresh_status()
        self._schedule_periodic_refresh()

    def _build_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#121316")
        header_frame.pack(fill="x", padx=18, pady=(16, 10))

        title = tk.Label(header_frame, text="KALSHI DESK CONTROL", font=("Segoe UI", 13, "bold"),
                         fg="#f0f2f5", bg="#121316")
        title.pack(side="left")

        self.live_badge = tk.Label(header_frame, text="● LIVE", font=("Segoe UI", 9, "bold"),
                                   fg="#0ca30c", bg="#121316")
        self.live_badge.pack(side="right")

        # Card 1: Account Metrics
        card_acc = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_acc.pack(fill="x", padx=18, pady=6)

        acc_inner = tk.Frame(card_acc, bg="#1c1e24")
        acc_inner.pack(fill="x", padx=14, pady=10)

        self.lbl_equity = tk.Label(acc_inner, text="$0.00", font=("Segoe UI", 20, "bold"),
                                   fg="#f0f2f5", bg="#1c1e24")
        self.lbl_equity.pack(anchor="w")

        self.lbl_split = tk.Label(acc_inner, text="Cash: $0.00  |  Positions: $0.00",
                                  font=("Segoe UI", 9), fg="#9ba1a6", bg="#1c1e24")
        self.lbl_split.pack(anchor="w", pady=(2, 4))

        self.lbl_pnl = tk.Label(acc_inner, text="Today: +$0.00  (Goal: $8.00)",
                                font=("Segoe UI", 10, "bold"), fg="#0ca30c", bg="#1c1e24")
        self.lbl_pnl.pack(anchor="w")

        # Card 2: Trading Bot Toggle
        card_bot = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_bot.pack(fill="x", padx=18, pady=6)

        bot_inner = tk.Frame(card_bot, bg="#1c1e24")
        bot_inner.pack(fill="x", padx=14, pady=10)

        bot_left = tk.Frame(bot_inner, bg="#1c1e24")
        bot_left.pack(side="left", fill="both", expand=True)

        tk.Label(bot_left, text="Trading Bot (15m Loop)", font=("Segoe UI", 10, "bold"),
                 fg="#f0f2f5", bg="#1c1e24").pack(anchor="w")
        self.lbl_bot_desc = tk.Label(bot_left, text="Checking status...", font=("Segoe UI", 8),
                                     fg="#9ba1a6", bg="#1c1e24")
        self.lbl_bot_desc.pack(anchor="w", pady=(2, 0))

        self.btn_bot_toggle = tk.Button(bot_inner, text="--", font=("Segoe UI", 9, "bold"),
                                        width=8, relief="flat", cursor="hand2", command=self._toggle_bot)
        self.btn_bot_toggle.pack(side="right")

        # Card 3: Clamshell / Lid Mode Toggle
        card_lid = tk.Frame(self.root, bg="#1c1e24", bd=0, highlightbackground="#2a2d36", highlightthickness=1)
        card_lid.pack(fill="x", padx=18, pady=6)

        lid_inner = tk.Frame(card_lid, bg="#1c1e24")
        lid_inner.pack(fill="x", padx=14, pady=10)

        lid_left = tk.Frame(lid_inner, bg="#1c1e24")
        lid_left.pack(side="left", fill="both", expand=True)

        tk.Label(lid_left, text="Run When Laptop Closed", font=("Segoe UI", 10, "bold"),
                 fg="#f0f2f5", bg="#1c1e24").pack(anchor="w")
        self.lbl_lid_desc = tk.Label(lid_left, text="Checking power plan...", font=("Segoe UI", 8),
                                     fg="#9ba1a6", bg="#1c1e24")
        self.lbl_lid_desc.pack(anchor="w", pady=(2, 0))

        self.btn_lid_toggle = tk.Button(lid_inner, text="--", font=("Segoe UI", 9, "bold"),
                                        width=8, relief="flat", cursor="hand2", command=self._toggle_lid)
        self.btn_lid_toggle.pack(side="right")

        # Action Buttons
        btn_frame = tk.Frame(self.root, bg="#121316")
        btn_frame.pack(fill="x", padx=18, pady=(12, 6))

        self.btn_run_now = tk.Button(btn_frame, text="⚡ Run Meeting & Cycle Now", font=("Segoe UI", 9, "bold"),
                                     bg="#2a78d6", fg="#ffffff", activebackground="#1f5bb5", activeforeground="#ffffff",
                                     relief="flat", cursor="hand2", height=2, command=self._run_cycle_now)
        self.btn_run_now.pack(fill="x", pady=(0, 6))

        self.btn_dashboard = tk.Button(btn_frame, text="📈 Open Live Dashboard", font=("Segoe UI", 9, "bold"),
                                       bg="#272a34", fg="#f0f2f5", activebackground="#343845", activeforeground="#ffffff",
                                       relief="flat", cursor="hand2", height=2, command=self._open_dashboard)
        self.btn_dashboard.pack(fill="x")

        # Footer
        self.lbl_footer = tk.Label(self.root, text="Ready", font=("Segoe UI", 8), fg="#6b7280", bg="#121316")
        self.lbl_footer.pack(side="bottom", pady=8)

    def _refresh_status(self):
        # 1. Account info
        total, cash, pos, today_pnl = get_account_info()
        self.lbl_equity.config(text=f"${total:.2f}")
        self.lbl_split.config(text=f"Cash: ${cash:.2f}  |  Positions: ${pos:.2f}")
        sign = "+" if today_pnl >= 0 else "-"
        pnl_color = "#0ca30c" if today_pnl >= 0 else "#e05656"
        self.lbl_pnl.config(text=f"Today: {sign}${abs(today_pnl):.2f}  (Goal: $8.00)", fg=pnl_color)

        # 2. Bot status
        bot_enabled, bot_state, next_run = get_bot_status()
        if bot_enabled:
            self.lbl_bot_desc.config(text=f"ACTIVE • Next cycle at {next_run}", fg="#0ca30c")
            self.btn_bot_toggle.config(text="ON", bg="#0ca30c", fg="#ffffff", activebackground="#098209")
        else:
            self.lbl_bot_desc.config(text="PAUSED • Task is disabled", fg="#e05656")
            self.btn_bot_toggle.config(text="OFF", bg="#3a3f4d", fg="#9ba1a6", activebackground="#2c303b")

        # 3. Lid status
        lid_awake = get_lid_status()
        if lid_awake:
            self.lbl_lid_desc.config(text="KEEP AWAKE • Won't sleep on lid close", fg="#0ca30c")
            self.btn_lid_toggle.config(text="ON", bg="#0ca30c", fg="#ffffff", activebackground="#098209")
        else:
            self.lbl_lid_desc.config(text="NORMAL SLEEP • Sleeps when lid closed", fg="#9ba1a6")
            self.btn_lid_toggle.config(text="OFF", bg="#3a3f4d", fg="#9ba1a6", activebackground="#2c303b")

    def _schedule_periodic_refresh(self):
        self.root.after(4000, self._auto_refresh)

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
        self.btn_run_now.config(state="disabled", text="⏳ Running Desk Cycle...")
        self.lbl_footer.config(text="Executing meeting, loop and orders...", fg="#2a78d6")

        def worker():
            try:
                subprocess.run([str(PY), str(RUN_LOOP_PY)], cwd=ROOT, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
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
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.lbl_footer.config(text="Dashboard launched at http://localhost:8501", fg="#0ca30c")
        except Exception as e:
            self.lbl_footer.config(text=f"Dashboard error: {e}", fg="#e05656")


def main():
    root = tk.Tk()
    app = KalshiControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
