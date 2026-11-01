"""Hourly job (run by Task Scheduler with pythonw.exe, so no window appears):
one paper-trading research cycle, one LLM desk cycle, one executor cycle (mode from
executor/.env), a ledger sync, and a notification digest. Output is appended to data/logs/loop.log."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = "logistic_l2-57cdc200f7@KXHIGHNY-h30"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STEPS = [
    ["-m", "predlab.cli", "loop", "--deployment", DEPLOYMENT, "--once"],
    ["-m", "desk.cli", "cycle"],         # LLM desk; a no-op until `desk.cli register`
    ["-m", "executor.cli", "run", "--model-version", DEPLOYMENT],
    ["-m", "executor.cli", "sync"],
    ["-m", "notify.cli", "digest"],      # phone alerts; a no-op until notify/.env is set
]


def main() -> int:
    log_path = ROOT / "data" / "logs" / "loop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worst = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        log.flush()
        for step in STEPS:
            r = subprocess.run([str(PY), *step], check=False, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=40 * 60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            log.write(f"--- {' '.join(step[1:3])} exit {r.returncode}\n")
            log.flush()
            worst = worst or r.returncode
    return worst


if __name__ == "__main__":
    sys.exit(main())
