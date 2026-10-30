"""Collector plumbing: time a run, record source health, and persist data-quality issues."""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy import Engine

from predlab.db.repo import record_health, record_issues
from predlab.logs import get_logger

log = get_logger(__name__)


@dataclass
class RunStats:
    source: str
    rows: dict[str, int] = field(default_factory=dict)
    issues: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def add(self, table: str, n: int) -> None:
        self.rows[table] = self.rows.get(table, 0) + n

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())


@asynccontextmanager
async def tracked_run(engine: Engine, source: str, *, reraise: bool = True):
    """Wrap a collection run. Health is recorded whether it succeeds or fails."""
    stats = RunStats(source)
    t0 = time.perf_counter()
    try:
        yield stats
    except Exception as e:  # recorded, then re-raised unless the caller opts out
        stats.error = f"{type(e).__name__}: {e}"
        record_health(engine, source, False, (time.perf_counter() - t0) * 1000, stats.total_rows,
                      stats.error)
        log.error("collector_failed", source=source, error=stats.error)
        if reraise:
            raise
    else:
        record_health(engine, source, True, (time.perf_counter() - t0) * 1000, stats.total_rows)
        log.info("collector_ok", source=source, rows=stats.rows, issues=stats.issues)


def persist_issues(engine: Engine, stats: RunStats, issues: list[dict]) -> None:
    if issues:
        stats.issues += record_issues(engine, issues)
