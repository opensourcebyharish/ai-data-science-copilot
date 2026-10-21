"""SQLAlchemy persistence for ML run history (Milestone 6).

Runs are saved to a SQLite database (``ml_runs.db``) in the project
root. The schema is created automatically on first use. Setting the
``ML_RUNS_DB_PATH`` environment variable moves the database (tests use
this to stay hermetic). Nothing here depends on Streamlit or the ML
engine — it is pure SQLAlchemy + stdlib.

Public API
----------
- ``get_engine()``         — shared SQLAlchemy engine (call once per process)
- ``init_db(engine)``      — create tables if they do not exist (+ migrate)
- ``save_run(result, ...)`` — insert a ModelResult, return the new run id
- ``load_runs()``          — return all runs as a list of dicts, newest first
- ``delete_run(run_id)``   — remove a single run by primary key
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

#: Database file lives at the project root alongside app.py.
_DB_PATH = Path(__file__).resolve().parents[1] / "ml_runs.db"

#: Module-level singleton engine (lazy-initialised) and the path it was
#: created for, so an ML_RUNS_DB_PATH override change takes effect.
_engine = None
_engine_path: Path | None = None


def _default_db_path() -> Path:
    """Where runs are stored unless ``ML_RUNS_DB_PATH`` overrides it.

    The override exists so tests (and any sandboxed deployment) can
    point the run history at a throwaway file without importing or
    reconfiguring anything — the app itself never sets it.
    """
    override = os.environ.get("ML_RUNS_DB_PATH")
    return Path(override) if override else _DB_PATH


class _Base(DeclarativeBase):
    pass


class MLRun(_Base):
    """One completed training run persisted to SQLite."""

    __tablename__ = "ml_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    task_type: str = Column(String(32), nullable=False)
    model_name: str = Column(String(64), nullable=False)
    target_column: str | None = Column(String(256), nullable=True)
    feature_columns: str = Column(Text, nullable=False)   # JSON array
    metrics: str = Column(Text, nullable=False)            # JSON object
    dataset_hash: str = Column(String(32), nullable=False, default="")
    #: Milestone 8: reproducibility settings (JSON object) — seeds,
    #: split sizes, preprocessing notes. Never raw rows or secrets.
    config: str = Column(Text, nullable=False, default="{}")


def get_engine(db_path: str | Path | None = None):
    """Return a SQLAlchemy engine for the project's run-history database.

    Without ``db_path`` the shared engine is returned, created on first
    call and reused for the process — unless ``ML_RUNS_DB_PATH`` points
    somewhere else, in which case the engine follows the configured
    path. With ``db_path`` a fresh, independent engine is returned
    instead: tests pass a temporary file and must never repoint the
    shared engine at it.
    """
    global _engine, _engine_path
    if db_path is not None:
        return create_engine(
            f"sqlite:///{Path(db_path)}", connect_args={"check_same_thread": False}
        )
    path = _default_db_path()
    if _engine is None or _engine_path != path:
        _engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False}
        )
        _engine_path = path
    return _engine


def init_db(engine=None) -> None:
    """Create the ``ml_runs`` table if it does not already exist.

    Existing databases created before Milestone 8 are upgraded in place
    by adding the ``config`` column when it is missing — a new column
    with a default, never a destructive rewrite.
    """
    eng = engine or get_engine()
    _Base.metadata.create_all(eng)
    _ensure_config_column(eng)


def _ensure_config_column(engine) -> None:
    """Add the ``config`` column to a pre-Milestone-8 database if needed."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if MLRun.__tablename__ not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns(MLRun.__tablename__)}
    if "config" in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE {MLRun.__tablename__} "
                "ADD COLUMN config TEXT NOT NULL DEFAULT '{}'"
            )
        )


def save_run(
    result,                          # core.models.ModelResult
    dataset_hash: str = "",
    engine=None,
    config: dict | None = None,
) -> int:
    """Persist a ModelResult to the database and return the new run id.

    Parameters
    ----------
    result:       A ``core.models.ModelResult`` instance.
    dataset_hash: Short hash of the source frame (for provenance).
    engine:       Optional engine override (defaults to the shared engine).
    config:       Optional reproducibility settings (JSON-serialisable
                  dict). Metadata only — never raw rows or secrets.
    """
    eng = engine or get_engine()
    init_db(eng)
    run = MLRun(
        created_at=datetime.now(timezone.utc),
        task_type=result.task_type,
        model_name=result.model_name,
        target_column=result.target_column,
        feature_columns=json.dumps(result.feature_columns),
        metrics=json.dumps(result.metrics),
        dataset_hash=dataset_hash,
        config=json.dumps(config or {}),
    )
    with Session(eng) as session:
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id


def load_runs(engine=None) -> list[dict[str, Any]]:
    """Return all saved runs as dicts, newest first.

    Returns an empty list when the table does not exist yet.
    """
    eng = engine or get_engine()
    init_db(eng)
    with Session(eng) as session:
        rows = session.query(MLRun).order_by(MLRun.id.desc()).all()
        return [_run_to_dict(row) for row in rows]


def delete_run(run_id: int, engine=None) -> bool:
    """Delete a run by primary key. Returns True if a row was removed."""
    eng = engine or get_engine()
    init_db(eng)
    with Session(eng) as session:
        row = session.get(MLRun, run_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_to_dict(row: MLRun) -> dict[str, Any]:
    """Convert an ORM row to a plain dict with deserialized JSON columns."""
    raw_config = getattr(row, "config", None) or "{}"
    try:
        config = json.loads(raw_config)
    except (TypeError, ValueError):
        config = {}
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "task_type": row.task_type,
        "model_name": row.model_name,
        "target_column": row.target_column,
        "feature_columns": json.loads(row.feature_columns),
        "metrics": json.loads(row.metrics),
        "dataset_hash": row.dataset_hash,
        "config": config,
    }
