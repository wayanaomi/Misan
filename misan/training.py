"""Reproducible training pipeline: dataset -> features -> models -> calibration ->
versioned artifact (PRD FR-28, FR-29; Section 11 modelling approach).

Usage: `python -m misan.cli train --dataset data/raw/<file>.parquet`
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from misan import ingestion
from misan.baseline import RuleBasedBaseline
from misan.config import SETTINGS, WINDOW_LENGTH_SAMPLES
from misan.detection import DetectionModel
from misan.diagnosis import DiagnosisModel
from misan.features import FEATURE_COLUMNS, build_feature_table
from misan.persistence import init_db, register_model_version
from misan.severity import SeverityModel

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
}


@dataclass
class TrainResult:
    version: str
    model_dir: Path
    chosen_model: str
    validation_macro_f1: float
    all_candidate_scores: dict[str, float]


def _group_split(df: pd.DataFrame, group_col: str, test_size: float, seed: int) -> tuple[pd.Index, pd.Index]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.index[train_idx], df.index[test_idx]


def prepare_features(raw_df: pd.DataFrame, window_length: int = WINDOW_LENGTH_SAMPLES) -> pd.DataFrame:
    clean_df, _ = ingestion.clean(raw_df)
    return build_feature_table(
        clean_df,
        window_length=window_length,
        step=max(window_length // 3, 1),
        group_col="run_id",
        passthrough_cols=("asset_id", "fault_class", "severity", "noise_level"),
    )


def _validate_training_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reject incompatible datasets before model training starts."""
    raw_df = ingestion.normalize_source_columns(raw_df)

    required_training = {"run_id", "fault_class", "severity"}
    missing = sorted(required_training - set(raw_df.columns))
    if missing:
        raise ValueError(
            "Dataset is not ready for training. Missing training columns: "
            + ", ".join(missing)
            + ". Use a compatible labelled training dataset. "
              "Historian files may still be imported for inspection."
        )

    problems = ingestion.validate_schema(raw_df)
    if problems:
        raise ValueError(
            "Dataset is incompatible with the configured sensor schema: "
            + "; ".join(problems)
            + ". Missing physical measurements will not be fabricated."
        )

    if raw_df.empty:
        raise ValueError("Training dataset is empty.")

    for column in ("run_id", "fault_class"):
        values = raw_df[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Training dataset contains empty {column} values.")

    severity = pd.to_numeric(raw_df["severity"], errors="coerce")
    if severity.isna().any() or not severity.map(pd.notna).all():
        raise ValueError("The severity column must contain numeric values.")
    if not severity.map(lambda value: abs(value) != float("inf")).all():
        raise ValueError("The severity column must contain finite values.")
    raw_df["severity"] = severity

    run_count = raw_df["run_id"].nunique()
    if run_count < 4:
        raise ValueError(
            f"Training requires at least 4 distinct run_id groups; found {run_count}."
        )

    if raw_df["fault_class"].nunique() < 2:
        raise ValueError("Training requires at least two distinct fault_class labels.")

    return raw_df


def train(dataset_path: str, seed: int = 42, test_size: float = 0.2, val_size: float = 0.15) -> TrainResult:
    raw_df = _validate_training_dataset(ingestion.read_raw(dataset_path))
    feature_df = prepare_features(raw_df)
    if feature_df.empty:
        raise ValueError("no windows could be built — dataset is shorter than the configured window length")

    train_groups, test_idx = _group_split(feature_df, "run_id", test_size, seed)
    train_df = feature_df.loc[train_groups]
    test_df = feature_df.loc[test_idx]
    fit_groups, val_idx = _group_split(train_df, "run_id", val_size, seed)
    fit_df = train_df.loc[fit_groups]
    val_df = train_df.loc[val_idx]

    # --- select the best diagnosis model family on the validation split ---
    scores: dict[str, float] = {}
    for name, estimator in CANDIDATE_MODELS.items():
        candidate = DiagnosisModel(estimator=_clone_estimator(estimator), calibrate=False)
        candidate.fit(fit_df, fit_df["fault_class"])
        preds = candidate.predict(val_df)
        scores[name] = float(f1_score(val_df["fault_class"], preds, average="macro", zero_division=0))
    best_name = max(scores, key=scores.get)

    # --- fit the chosen model, calibrated, on train+val (not test) ---
    diagnosis_model = DiagnosisModel(estimator=_clone_estimator(CANDIDATE_MODELS[best_name]), calibrate=True)
    diagnosis_model.fit(train_df, train_df["fault_class"])

    detection_model = DetectionModel()
    detection_model.fit(train_df, train_df["fault_class"])

    severity_model = SeverityModel()
    faulty_train = train_df[train_df["fault_class"] != "NORMAL"]
    if len(faulty_train) >= 10:
        severity_model.fit(faulty_train, faulty_train["severity"].astype(float))

    healthy_train = train_df[train_df["fault_class"] == "NORMAL"]
    baseline_mean = healthy_train[FEATURE_COLUMNS].mean() if len(healthy_train) else None
    baseline_std = healthy_train[FEATURE_COLUMNS].std() if len(healthy_train) else None

    rule_baseline = RuleBasedBaseline()
    if len(healthy_train):
        rule_baseline.fit(healthy_train[FEATURE_COLUMNS])

    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    model_dir = SETTINGS.models_dir / version
    model_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_model.save(str(model_dir / "diagnosis_model.joblib"))
    detection_model.save(str(model_dir / "detection_model.joblib"))
    severity_model.save(str(model_dir / "severity_model.joblib"))
    joblib.dump(rule_baseline, model_dir / "rule_baseline.joblib")
    joblib.dump(baseline_mean, model_dir / "baseline_mean.joblib")
    joblib.dump(baseline_std, model_dir / "baseline_std.joblib")
    test_df.to_parquet(model_dir / "test_features.parquet")

    metadata = {
        "version": version,
        "chosen_model": best_name,
        "candidate_validation_macro_f1": scores,
        "source_dataset": str(dataset_path),
        "n_train_windows": len(train_df),
        "n_test_windows": len(test_df),
        "severity_model_trained": severity_model.is_trained,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    init_db()
    register_model_version(version=version, model_type=best_name, metrics={"validation_macro_f1": scores[best_name]})
    _write_latest_pointer(version)

    return TrainResult(
        version=version,
        model_dir=model_dir,
        chosen_model=best_name,
        validation_macro_f1=scores[best_name],
        all_candidate_scores=scores,
    )


def _clone_estimator(estimator):
    from sklearn.base import clone

    return clone(estimator)


def _write_latest_pointer(version: str) -> None:
    (SETTINGS.models_dir / "latest.json").write_text(json.dumps({"version": version}))


def load_latest_version() -> str | None:
    pointer = SETTINGS.models_dir / "latest.json"
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text())["version"]
