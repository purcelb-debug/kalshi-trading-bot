"""Persistent executor state, hash-chained audit log, kill switch, and pre-trade risk checks."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from executor.config import ExecutorSettings


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class State:
    day: str = ""
    day_start_value_usd: float | None = None
    orders_today: int = 0
    consecutive_errors: int = 0
    halted: str = ""                      # non-empty = halted, with the reason
    history: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "State":
        if path.is_file():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)

    def roll_day(self, total_value_usd: float) -> None:
        today = _now().date().isoformat()
        if self.day != today:
            self.day, self.day_start_value_usd, self.orders_today = today, total_value_usd, 0


class AuditLog:
    """Append-only JSONL; each line carries sha256(prev_hash + payload)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.is_file():
            return "0" * 64
        last = ""
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else "0" * 64

    def write(self, event: str, **data) -> dict:
        prev = self._last_hash()
        payload = {"ts": _now().isoformat(), "event": event, **data}
        body = json.dumps(payload, sort_keys=True, default=str)
        rec = {**payload, "prev": prev, "hash": hashlib.sha256((prev + body).encode()).hexdigest()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def verify(self) -> bool:
        prev = "0" * 64
        if not self.path.is_file():
            return True
        for line in self.path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            h, p = rec.pop("hash"), rec.pop("prev")
            body = json.dumps(rec, sort_keys=True, default=str)
            if p != prev or hashlib.sha256((prev + body).encode()).hexdigest() != h:
                return False
            prev = h
        return True


def kill_switch_on(cfg: ExecutorSettings) -> bool:
    return Path(cfg.kill_switch_path).exists()


@dataclass(frozen=True)
class Check:
    ok: bool
    reason: str = ""


def startup_checks(cfg: ExecutorSettings, state: State) -> Check:
    if kill_switch_on(cfg):
        return Check(False, f"kill switch present at {cfg.kill_switch_path}")
    if state.halted:
        return Check(False, f"halted: {state.halted} (clear with `executor resume`)")
    if cfg.mode in ("demo", "live"):
        missing = cfg.missing_limits()
        if missing:
            return Check(False, f"money limits not configured: {', '.join(missing)}")
        key_id, key_path = cfg.credentials()
        if not key_id or not key_path:
            return Check(False, f"{cfg.mode} credentials not configured")
    return Check(True)


def pre_trade_check(cfg: ExecutorSettings, state: State, *, cost_usd: float, cash_usd: float,
                    open_exposure_usd: float, market_exposure_usd: float,
                    total_value_usd: float) -> Check:
    """Every limit is re-checked immediately before each individual order."""
    if kill_switch_on(cfg):
        return Check(False, "kill switch present")
    if state.halted:
        return Check(False, f"halted: {state.halted}")
    if state.orders_today >= cfg.max_orders_per_day:
        return Check(False, f"max orders per day reached ({cfg.max_orders_per_day})")
    if state.consecutive_errors >= cfg.max_consecutive_errors:
        return Check(False, "too many consecutive errors")
    if cfg.mode == "dry_run":
        return Check(True)
    if cost_usd > cfg.max_order_usd:
        return Check(False, f"order ${cost_usd:.2f} > max_order_usd ${cfg.max_order_usd:.2f}")
    if open_exposure_usd + cost_usd > cfg.bankroll_cap_usd:
        return Check(False, f"exposure would be ${open_exposure_usd + cost_usd:.2f} > "
                            f"bankroll cap ${cfg.bankroll_cap_usd:.2f}")
    if cfg.max_market_usd is not None and market_exposure_usd + cost_usd > cfg.max_market_usd:
        return Check(False, "per-market cap exceeded")
    if cost_usd > cash_usd:
        return Check(False, "insufficient cash")
    if state.day_start_value_usd is not None and \
            state.day_start_value_usd - total_value_usd >= cfg.daily_loss_stop_usd:
        return Check(False, f"daily loss stop hit (down "
                            f"${state.day_start_value_usd - total_value_usd:.2f})")
    return Check(True)
