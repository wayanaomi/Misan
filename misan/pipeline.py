"""The live Misan engine: raw window -> clean -> features -> detect -> diagnose ->
severity -> explain -> physics-validate -> structured record (PRD Sections 2, 7, 8).

Model artifacts are loaded lazily from the active trained version. If no model has
been trained yet, `diagnose_window` raises `ModelNotTrainedError` rather than ever
fabricating a diagnosis — honest empty state per PRD Section 3.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import pandas as pd

from misan import ingestion
from misan.config import FAULT_BY_CODE, SETTINGS, WINDOW_LENGTH_SAMPLES
from misan.detection import DetectionModel, PersistenceTracker
from misan.diagnosis import DiagnosisModel, class_probabilities, top_class
from misan.explainability import explain, to_sentence
from misan.features import window_feature_vector
from misan.persistence import get_active_model_version, record_diagnostic
from misan.physics_validation import resolve_with_fallback
from misan.schemas import DiagnosticRecord, SensorWindow
from misan.severity import SeverityModel, SeverityTrendTracker, severity_label
from misan.training import load_latest_version


class ModelNotTrainedError(RuntimeError):
    pass


class InsufficientDataError(ValueError):
    def __init__(self, notes: list[str]):
        super().__init__("; ".join(notes))
        self.notes = notes


@dataclass
class ModelBundle:
    version: str
    diagnosis_model: DiagnosisModel
    detection_model: DetectionModel
    severity_model: SeverityModel
    baseline_mean: pd.Series | None
    baseline_std: pd.Series | None


_bundle_cache: ModelBundle | None = None
_persistence_tracker = PersistenceTracker(n=SETTINGS.persistence_n)
_severity_tracker = SeverityTrendTracker()


def _resolve_active_version() -> str | None:
    active = get_active_model_version()
    if active:
        return active["version"]
    return load_latest_version()


def get_active_bundle(force_reload: bool = False) -> ModelBundle | None:
    global _bundle_cache
    version = _resolve_active_version()
    if version is None:
        return None
    if not force_reload and _bundle_cache is not None and _bundle_cache.version == version:
        return _bundle_cache

    model_dir = SETTINGS.models_dir / version
    if not model_dir.exists():
        return None
    _bundle_cache = ModelBundle(
        version=version,
        diagnosis_model=DiagnosisModel.load(str(model_dir / "diagnosis_model.joblib")),
        detection_model=DetectionModel.load(str(model_dir / "detection_model.joblib")),
        severity_model=SeverityModel.load(str(model_dir / "severity_model.joblib")),
        baseline_mean=joblib.load(model_dir / "baseline_mean.joblib"),
        baseline_std=joblib.load(model_dir / "baseline_std.joblib"),
    )
    return _bundle_cache


def diagnose_window(window: SensorWindow, confidence_threshold: float | None = None) -> DiagnosticRecord:
    bundle = get_active_bundle()
    if bundle is None:
        raise ModelNotTrainedError("no trained model is available yet — run `python -m misan.cli train` first")

    threshold = confidence_threshold if confidence_threshold is not None else SETTINGS.confidence_threshold

    raw_df = ingestion.sensor_window_to_frame(window.asset_id, window.timestamps, window.readings)
    clean_df, _ = ingestion.clean(raw_df)
    window_df = clean_df.tail(WINDOW_LENGTH_SAMPLES)
    quality = ingestion.window_quality(window_df, WINDOW_LENGTH_SAMPLES)
    if not quality.sufficient:
        raise InsufficientDataError(quality.notes)

    try:
        feature_row = pd.Series(window_feature_vector(window_df), dtype=float)
    except (ValueError, KeyError) as exc:
        raise InsufficientDataError([str(exc)]) from exc

    import numpy as np
    X = feature_row.to_frame().T
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise InsufficientDataError([
            "diagnostic features contain missing or non-finite values; "
            "check required sensor readings and units"
        ])

    is_faulty_pred = bool(bundle.detection_model.predict_is_faulty(X)[0])
    alert = _persistence_tracker.update(window.asset_id, is_faulty_pred)

    proba_row = bundle.diagnosis_model.predict_proba_table(X).iloc[0]
    probabilities = class_probabilities(proba_row)
    top_code, top_conf = top_class(proba_row)

    if not is_faulty_pred:
        fault_class = "NORMAL"
    elif top_conf < threshold:
        fault_class = "UNKNOWN"
    else:
        fault_class = top_code

    # FR-23: if the top diagnosis fails the physics signature check, try the next
    # most probable class that passes it, and record that the fallback happened.
    check_target = top_code if fault_class not in ("NORMAL",) else "NORMAL"
    resolved_class, status, note, fallback_used = resolve_with_fallback(
        probabilities, bundle.baseline_mean, feature_row, top_class=check_target
    )
    if fallback_used and fault_class not in ("NORMAL", "UNKNOWN"):
        fault_class = resolved_class

    fault_meta = FAULT_BY_CODE.get(fault_class)
    component = fault_meta.component if fault_meta else "unknown"
    group = fault_meta.group if fault_meta else "unknown"

    severity_index: float | None = None
    severity_lvl: str | None = None
    severity_trend = "unknown"
    if fault_class not in ("NORMAL", "UNKNOWN") and bundle.severity_model.is_trained:
        severity_index = float(bundle.severity_model.predict(X)[0])
        severity_lvl = severity_label(severity_index)
        severity_trend = _severity_tracker.update(window.asset_id, severity_index)

    if fault_class == "NORMAL":
        top_features: list = []
        explanation = "Operation is within the learned healthy range."
    else:
        top_features = explain(bundle.diagnosis_model, X, predicted_class=top_code)
        explanation = to_sentence(top_features)

    record = DiagnosticRecord(
        record_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        asset_id=window.asset_id,
        fault_detected=bool(alert and fault_class not in ("NORMAL", "UNKNOWN")),
        fault_class=fault_class,
        component=component,
        group=group,
        class_probabilities=probabilities,
        confidence=top_conf,
        severity_index=severity_index,
        severity_level=severity_lvl,
        severity_trend=severity_trend,
        top_features=top_features,
        explanation=explanation,
        validation_status=status,
        validation_note=note,
        model_version=bundle.version,
        data_quality=quality,
    )
    record_diagnostic(record)
    return record
