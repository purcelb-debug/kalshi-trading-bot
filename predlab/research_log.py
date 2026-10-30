"""Append-only research log in the simulation database (promotions, cycles, alerts)."""

from sqlalchemy import Engine

from predlab.db.schema import research_log
from predlab.timeutil import utcnow


def log_event(sim_engine: Engine, component: str, level: str, message: str,
              payload: dict | None = None, ts=None) -> None:
    with sim_engine.begin() as c:
        c.execute(research_log.insert().values(ts=ts or utcnow(), component=component,
                                               level=level, message=message, payload=payload))
