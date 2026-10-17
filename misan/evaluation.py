"""Benchmark evaluation: per-class metrics, calibration error, noise robustness,
detection delay, false-alarm rate, baseline comparison and physics-check rates
(PRD FR-32, Section 11, Section 14 evaluation plan).

Usage: `python -m misan.cli evaluate --version <model_version>`
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from misan.config import FAULT_CODES, SETTINGS
from misan.detection import DetectionModel, PersistenceTracker
from misan.diagnosis import DiagnosisModel
from misan.features import FEATURE_COLUMNS
from misan.persistence import save_evaluation_run
from misan.physics_validation import check_signature
from misan.schemas import EvaluationSummary


def _expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_conf - bin_acc)
    return float(ece)


def _detection_delay(test_df: pd.DataFrame, predictions: np.ndarray, persistence_n: int) -> float | None:
    df = test_df.copy()
    df["predicted_faulty"] = np.asarray(predictions, dtype=bool)
    delays = []

    for run_id, group in df.groupby("run_id"):
        if (group["fault_class"] == "NORMAL").all():
            continue

        group = group.sort_values("timestamp").reset_index(drop=True)
        tracker = PersistenceTracker(n=persistence_n)

        for i, is_faulty in enumerate(group["predicted_faulty"]):
            if tracker.update(str(run_id), bool(is_faulty)):
                delays.append(i)
                break

    # Runs without an alert are misses, not long detection delays.
    return float(np.mean(delays)) if delays else None


def _detection_run_metrics(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    persistence_n: int,
) -> dict[str, float | int]:
    """Measure alert coverage across faulty runs, not individual windows."""
    df = test_df.copy()
    df["predicted_faulty"] = np.asarray(predictions, dtype=bool)

    faulty_runs = 0
    detected_runs = 0

    for run_id, group in df.groupby("run_id"):
        if (group["fault_class"] == "NORMAL").all():
            continue

        faulty_runs += 1
        group = group.sort_values("timestamp")
        tracker = PersistenceTracker(n=persistence_n)

        for is_faulty in group["predicted_faulty"]:
            if tracker.update(str(run_id), bool(is_faulty)):
                detected_runs += 1
                break

    miss_rate = (
        (faulty_runs - detected_runs) / faulty_runs
        if faulty_runs else 0.0
    )
    run_recall = detected_runs / faulty_runs if faulty_runs else 0.0

    return {
        "detection_miss_rate": float(miss_rate),
        "detection_run_recall": float(run_recall),
        "n_faulty_runs": faulty_runs,
        "n_detected_faulty_runs": detected_runs,
    }


def evaluate(version: str, persistence_n: int = 3) -> EvaluationSummary:
    model_dir = SETTINGS.models_dir / version
    diagnosis_model: DiagnosisModel = DiagnosisModel.load(str(model_dir / "diagnosis_model.joblib"))
    detection_model: DetectionModel = DetectionModel.load(str(model_dir / "detection_model.joblib"))
    baseline_mean = joblib.load(model_dir / "baseline_mean.joblib")
    baseline_std = joblib.load(model_dir / "baseline_std.joblib")
    rule_baseline = joblib.load(model_dir / "rule_baseline.joblib")
    test_df = pd.read_parquet(model_dir / "test_features.parquet")

    y_true = test_df["fault_class"].to_numpy()
    proba_table = diagnosis_model.predict_proba_table(test_df)
    y_pred = proba_table.idxmax(axis=1).to_numpy()
    confidence = proba_table.max(axis=1).to_numpy()

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=FAULT_CODES, average=None, zero_division=0
    )
    per_class = {
        code: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for code, p, r, f in zip(FAULT_CODES, precision, recall, f1)
    }
    cm = confusion_matrix(y_true, y_pred, labels=FAULT_CODES)
    confusion = {FAULT_CODES[i]: {FAULT_CODES[j]: int(cm[i, j]) for j in range(len(FAULT_CODES))} for i in range(len(FAULT_CODES))}

    healthy_mask = y_true == "NORMAL"
    false_alarm_rate = float((y_pred[healthy_mask] != "NORMAL").mean()) if healthy_mask.any() else 0.0

    correct = (y_true == y_pred).astype(float)
    calibration_error = _expected_calibration_error(confidence, correct)

    detector_predictions = detection_model.predict_is_faulty(test_df[FEATURE_COLUMNS])
    detection_delay = _detection_delay(
        test_df, detector_predictions, persistence_n
    )
    detection_metrics = _detection_run_metrics(
        test_df, detector_predictions, persistence_n
    )

    noise_robustness: dict[str, float] = {}
    if "noise_level" in test_df.columns:
        for level, group in test_df.groupby("noise_level"):
            group_pred = diagnosis_model.predict(group)
            noise_robustness[str(level)] = float(
                f1_score(group["fault_class"], group_pred, average="macro", zero_division=0)
            )

    baseline_pred = rule_baseline.predict(test_df[FEATURE_COLUMNS])
    baseline_macro_f1 = float(f1_score(y_true, baseline_pred, average="macro", zero_division=0))
    baseline_pred_arr = np.array(baseline_pred)
    baseline_false_alarm_rate = (
        float((baseline_pred_arr[healthy_mask] != "NORMAL").mean()) if healthy_mask.any() else 0.0
    )

    confirmed = rejected = 0
    for i in range(len(test_df)):
        status, _ = check_signature(test_df.iloc[i][FEATURE_COLUMNS], y_pred[i], baseline_mean, baseline_std)
        if status == "confirmed":
            confirmed += 1
        elif status == "rejected":
            rejected += 1
    n = len(test_df) or 1
    physics_confirmation_rate = confirmed / n
    physics_rejection_rate = rejected / n

    single_row = test_df.iloc[[0]]
    start = time.perf_counter()
    for _ in range(50):
        diagnosis_model.predict_proba_table(single_row)
    inference_time_ms = (time.perf_counter() - start) / 50 * 1000

    summary = EvaluationSummary(
        model_version=version,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=confusion,
        false_alarm_rate=false_alarm_rate,
        detection_delay_windows=detection_delay,
        **detection_metrics,
        calibration_error=calibration_error,
        inference_time_ms=float(inference_time_ms),
        noise_robustness=noise_robustness,
        baseline_macro_f1=baseline_macro_f1,
        baseline_false_alarm_rate=baseline_false_alarm_rate,
        physics_confirmation_rate=physics_confirmation_rate,
        physics_rejection_rate=physics_rejection_rate,
        n_test_windows=len(test_df),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    save_evaluation_run(version, json.loads(summary.model_dump_json()))
    report_path = SETTINGS.data_dir / "reports" / f"evaluation_{version}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary.model_dump_json(indent=2))
    return summary
