"""FastAPI service exposing Misan's diagnostic engine, dataset registry and model/
evaluation metadata (PRD FR-24, FR-25; Section 10 System Context)."""
from __future__ import annotations

import shutil
import hmac
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from misan import ingestion, training
from misan.config import FAULT_CLASSES, SETTINGS
from misan.evaluation import evaluate as run_evaluation
from misan.persistence import (
    get_active_model_version,
    init_db,
    list_alerts,
    list_datasets,
    list_diagnostics,
    list_evaluation_runs,
    register_dataset,
)
from misan.pipeline import InsufficientDataError, ModelNotTrainedError, diagnose_window
from misan.schemas import DiagnosticRecord, SensorWindow


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    yield


def require_api_key(x_misan_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MISAN_API_KEY", "")
    if os.environ.get("MISAN_ENV", "").lower() == "production" and not expected:
        raise HTTPException(status_code=503, detail="Server security is not configured")
    # No key is required in local development; production must configure one.
    if expected and (
        not x_misan_api_key
        or not hmac.compare_digest(x_misan_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(
    title="Misan",
    description="AI fault detection and diagnosis module for HVAC systems",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict:
    active = get_active_model_version()
    version = active["version"] if active else training.load_latest_version()
    return {
        "status": "ok",
        "active_model_version": version,
    }

@app.get("/faults")
def faults() -> list[dict]:
    return [f.__dict__ for f in FAULT_CLASSES]


@app.post("/diagnose", response_model=DiagnosticRecord, dependencies=[Depends(require_api_key)])
def diagnose(window: SensorWindow) -> DiagnosticRecord:
    try:
        return diagnose_window(window)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail={"message": "insufficient or invalid sensor data", "notes": exc.notes}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/alerts/recent")
def recent_alerts(limit: int = 20) -> list[dict]:
    return list_alerts(limit=limit)


@app.get("/diagnostics/recent")
def recent_diagnostics(limit: int = 20) -> list[dict]:
    return list_diagnostics(limit=limit)


@app.get("/models/active")
def active_model() -> dict | None:
    active = get_active_model_version()
    if active:
        return active

    version = training.load_latest_version()
    if version is None:
        return None

    return {
        "version": version,
        "model_type": "trained",
        "created_at": None,
        "metrics": {},
    }

@app.get("/models/evaluations")
def evaluations(limit: int = 10) -> list[dict]:
    return list_evaluation_runs(limit=limit)


@app.get("/datasets")
def datasets() -> list[dict]:
    return list_datasets()


@app.post("/datasets/import", dependencies=[Depends(require_api_key)])
async def import_dataset(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "dataset.csv").suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise HTTPException(status_code=422, detail="only .csv or .parquet files are accepted")

    # Ignore any directory components supplied in the uploaded filename.
    safe_name = Path(file.filename or "dataset.csv").name
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=422, detail="invalid filename")

    dest_dir = SETTINGS.data_dir / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}_{safe_name}"

    # Enforce a 25 MB upload limit while streaming to disk.
    total = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 25 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="file exceeds 25 MB limit")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        raw_df = ingestion.read_raw(dest)
        problems = ingestion.validate_schema(raw_df)
        if problems:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail={"message": "schema validation failed", "problems": problems})
        _, quality = ingestion.clean(raw_df)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    dataset_id = register_dataset(name=safe_name, path=str(dest), source="imported", n_rows=len(raw_df))
    return JSONResponse({"dataset_id": dataset_id, "n_rows": len(raw_df), "data_quality": quality.model_dump()})


@app.post("/train", dependencies=[Depends(require_api_key)])
def train_endpoint(dataset_path: str) -> dict:
    try:
        result = training.train(dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "version": result.version,
        "chosen_model": result.chosen_model,
        "validation_macro_f1": result.validation_macro_f1,
        "all_candidate_scores": result.all_candidate_scores,
    }


@app.post("/evaluate", dependencies=[Depends(require_api_key)])
def evaluate_endpoint(version: str | None = None) -> dict:
    target_version = version or training.load_latest_version()
    if not target_version:
        raise HTTPException(status_code=503, detail="no trained model is available yet")
    summary = run_evaluation(target_version)
    return summary.model_dump()
