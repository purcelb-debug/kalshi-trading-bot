"""Hourly job (run by Task Scheduler with pythonw.exe, so no window appears):
for every quant deployment one paper-trading research cycle, then one LLM desk cycle over
every desk deployment, then one executor cycle per quant deployment (mode from
executor/.env; the safety caps are shared across them), a ledger sync, and a notification
digest. Output is appended to data/logs/loop.log."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS = [
    "logistic_l2-57cdc200f7@KXHIGHNY-h30",
]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STEPS = (
    [["-m", "predlab.cli", "loop", "--deployment", d, "--once"] for d in DEPLOYMENTS]
    + [["-m", "desk.cli", "cycle", "--all"]]    # LLM desk; a no-op until `desk.cli register`
    + [["-m", "executor.cli", "run", "--model-version", d] for d in DEPLOYMENTS]
    + [["-m", "executor.cli", "sync"],
       ["-m", "notify.cli", "digest"]]          # phone alerts; a no-op until notify/.env is set
)


def main() -> int:
    log_path = ROOT / "data" / "logs" / "loop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worst = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        log.flush()
        for step in STEPS:
            started = datetime.now()
            try:
                code = subprocess.run([str(PY), *step], check=False, cwd=ROOT, stdout=log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      timeout=20 * 60,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                      ).returncode
            except subprocess.TimeoutExpired:
                code = 124                  # one stuck market must not block the rest
            took = (datetime.now() - started).total_seconds()
            log.write(f"--- {' '.join(step[1:])} exit {code} ({took:.0f}s)\n")
            log.flush()
            worst = worst or code
    return worst


if __name__ == "__main__":
    sys.exit(main())
