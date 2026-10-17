"""SQLite-backed operational store: diagnostic records, dataset registry, model
registry and evaluation runs. The database starts empty — nothing here is seeded
(PRD Section 3, "zero demo data")."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from misan.config import SETTINGS
from misan.schemas import DiagnosticRecord


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    source = Column(String, nullable=False)  # "imported" | "simulated"
    n_rows = Column(Integer, nullable=False, default=0)
    imported_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    notes = Column(Text, default="")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, nullable=False, unique=True)
    model_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metrics_json = Column(Text, default="{}")
    is_active = Column(Boolean, default=False)


class DiagnosticRecordRow(Base):
    __tablename__ = "diagnostic_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    payload_json = Column(JSON, nullable=False)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    summary_json = Column(JSON, nullable=False)


_engine = create_engine(f"sqlite:///{SETTINGS.db_path}", future=True)
SessionLocal = sessionmaker(bind=_engine, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    return SessionLocal()


def record_diagnostic(record: DiagnosticRecord) -> None:
    with get_session() as session:
        session.add(DiagnosticRecordRow(record_id=record.record_id, payload_json=json.loads(record.model_dump_json())))
        session.commit()


def list_diagnostics(limit: int = 20) -> list[dict]:
    if limit <= 0:
        return []
    with get_session() as session:
        rows = (
            session.query(DiagnosticRecordRow)
            .order_by(DiagnosticRecordRow.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.payload_json for r in rows]


def list_alerts(limit: int = 20) -> list[dict]:
    """Return only records where the diagnostic persistence rule raised an alert."""
    if limit <= 0:
        return []
    with get_session() as session:
        rows = (
            session.query(DiagnosticRecordRow)
            .order_by(DiagnosticRecordRow.created_at.desc())
            .all()
        )
        alerts = []
        for row in rows:
            payload = row.payload_json
            if payload.get("fault_detected") is True:
                alerts.append(payload)
                if len(alerts) >= limit:
                    break
        return alerts


def register_dataset(name: str, path: str, source: str, n_rows: int, notes: str = "") -> int:
    with get_session() as session:
        ds = Dataset(name=name, path=path, source=source, n_rows=n_rows, notes=notes)
        session.add(ds)
        session.commit()
        return ds.id


def list_datasets() -> list[dict]:
    with get_session() as session:
        rows = session.query(Dataset).order_by(Dataset.imported_at.desc()).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "path": r.path,
                "source": r.source,
                "n_rows": r.n_rows,
                "imported_at": r.imported_at.isoformat() if r.imported_at else None,
                "notes": r.notes,
            }
            for r in rows
        ]


def register_model_version(version: str, model_type: str, metrics: dict) -> None:
    with get_session() as session:
        session.query(ModelVersion).update({ModelVersion.is_active: False})
        session.add(
            ModelVersion(version=version, model_type=model_type, metrics_json=json.dumps(metrics), is_active=True)
        )
        session.commit()


def get_active_model_version() -> dict | None:
    with get_session() as session:
        row = session.query(ModelVersion).filter_by(is_active=True).order_by(ModelVersion.created_at.desc()).first()
        if not row:
            return None
        return {
            "version": row.version,
            "model_type": row.model_type,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "metrics": json.loads(row.metrics_json),
        }


def save_evaluation_run(model_version: str, summary: dict) -> None:
    with get_session() as session:
        session.add(EvaluationRun(model_version=model_version, summary_json=summary))
        session.commit()


def list_evaluation_runs(limit: int = 10) -> list[dict]:
    with get_session() as session:
        rows = session.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit).all()
        return [
            {"model_version": r.model_version, "created_at": r.created_at.isoformat(), "summary": r.summary_json}
            for r in rows
        ]
