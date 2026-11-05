"""Hourly job (run by Task Scheduler with pythonw.exe, so no window appears):
for every quant deployment (LIVE and PAPER_ONLY) one paper-trading research cycle, then one LLM desk cycle over
every desk deployment, then one executor cycle per LIVE deployment (mode from
executor/.env; the safety caps are shared across them), a ledger sync, and a notification
digest. Output is appended to data/logs/loop.log."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = [                     # paper loop AND the executor (real money in live mode)
    "logistic_l2-57cdc200f7@KXHIGHNY-h30",
    "market_implied+platt-5e74e39fb3@KXHIGHCHI-h30",

]
PAPER_ONLY = [               # paper loop only: no model here has beaten the market yet

    "market_implied+platt-5e74e39fb3@KXHIGHMIA-h30",
]
DESK_LIVE = [                # LLM desk deployments the executor trades (exits + buys)
    "desk@ALL-h48",
]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STEPS = (
    [["-m", "predlab.cli", "loop", "--deployment", d, "--once"] for d in LIVE + PAPER_ONLY]
    + [["-m", "desk.cli", "cycle", "--all"]]    # LLM desk; a no-op until `desk.cli register`
    + [["-m", "executor.cli", "run", "--model-version", d] for d in LIVE + DESK_LIVE]
    + [["-m", "executor.cli", "sync"],
       ["-m", "notify.cli", "digest"]]          # phone alerts; a no-op until notify/.env is set
)


def _try_lock(fh) -> bool:
    """Non-blocking OS lock, released by the OS if this process dies."""
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def main() -> int:
    log_path = ROOT / "data" / "logs" / "loop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # One job at a time (Task Scheduler, the control panel's run button, a manual run):
    # two at once hold duplicate meetings and race each other's orders.
    lock = (ROOT / "data" / "run_loop.lock").open("a+b")
    if not _try_lock(lock):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} skipped: another run is "
                      "still going =====\n")
        return 0
    worst = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        log.flush()
        for step in STEPS:
            started = datetime.now()
            p = subprocess.Popen([str(PY), *step], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                code = p.wait(timeout=(40 if step[1] == "desk.cli" else 20) * 60)
            except subprocess.TimeoutExpired:
                # The venv python.exe is a launcher with the real interpreter as its child;
                # killing only the launcher once left an orphan holding up later runs.
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], check=False,
                               capture_output=True)
                p.wait()
                code = 124                  # one stuck market must not block the rest
            took = (datetime.now() - started).total_seconds()
            log.write(f"--- {' '.join(step[1:])} exit {code} ({took:.0f}s)\n")
            log.flush()
            worst = worst or code
    return worst


if __name__ == "__main__":
    sys.exit(main())
