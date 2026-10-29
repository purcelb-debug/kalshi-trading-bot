"""Engine creation, idempotent upserts, and health / data-quality bookkeeping."""

from collections.abc import Iterable, Sequence

from sqlalchemy import Engine, Table, create_engine, event, select
from sqlalchemy.dialects import postgresql, sqlite

from predlab.db.schema import MARKET_MD, SIM_MD, data_quality_issues, source_health
from predlab.timeutil import utcnow

_CHUNK = 5000


def make_engine(url: str) -> Engine:
    engine = create_engine(url, future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def init_market_db(engine: Engine) -> None:
    MARKET_MD.create_all(engine)


def init_sim_db(engine: Engine) -> None:
    SIM_MD.create_all(engine)


def _insert_for(engine: Engine):
    if engine.dialect.name == "sqlite":
        return sqlite.insert
    if engine.dialect.name == "postgresql":
        return postgresql.insert
    raise NotImplementedError(f"upsert not supported for dialect {engine.dialect.name}")


def upsert(engine: Engine, table: Table, rows: Iterable[dict], keys: Sequence[str],
           update: bool = True) -> int:
    """Insert rows; on key conflict update non-key columns (or skip if update=False).

    Rows are deduplicated on ``keys`` within the batch (last one wins) before writing.
    Returns the number of distinct rows submitted.
    """
    dedup: dict[tuple, dict] = {}
    for r in rows:
        dedup[tuple(r[k] for k in keys)] = r
    if not dedup:
        return 0
    batch = list(dedup.values())
    stmt = _insert_for(engine)(table)
    cols = [c for c in batch[0] if c not in keys]
    if update and cols:
        stmt = stmt.on_conflict_do_update(
            index_elements=list(keys), set_={c: stmt.excluded[c] for c in cols}
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(keys))
    with engine.begin() as conn:
        for i in range(0, len(batch), _CHUNK):
            conn.execute(stmt, batch[i : i + _CHUNK])  # executemany with one compiled statement
    return len(batch)


def record_health(engine: Engine, source: str, ok: bool, latency_ms: float, rows: int,
                  error: str | None = None) -> None:
    now = utcnow()
    with engine.begin() as conn:
        prev = conn.execute(
            select(source_health.c.consecutive_failures, source_health.c.last_success)
            .where(source_health.c.source == source)
        ).first()
    failures = 0 if ok else ((prev.consecutive_failures or 0) + 1 if prev else 1)
    upsert(engine, source_health, [{
        "source": source,
        "last_attempt": now,
        "last_success": now if ok else (prev.last_success if prev else None),
        "last_error": None if ok else (error or "unknown error")[:2000],
        "last_latency_ms": latency_ms,
        "rows_last_run": rows,
        "consecutive_failures": failures,
    }], keys=["source"])


def record_issues(engine: Engine, issues: Iterable[dict]) -> int:
    now = utcnow()
    rows = [{**i, "detected_at": now} for i in issues]
    return upsert(engine, data_quality_issues, rows,
                  keys=["source", "entity", "entity_id", "check_name"])
