"""Pydantic contracts shared by the API, persistence layer and dashboard.
`DiagnosticRecord` mirrors the output schema in PRD Section 8 exactly."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from misan.config import SENSOR_NAMES, WINDOW_LENGTH_SAMPLES

ValidationStatus = Literal["confirmed", "rejected", "unchecked"]
SeverityLevel = Literal["low", "medium", "high"]
SeverityTrend = Literal["rising", "stable", "falling", "unknown"]


class SensorWindow(BaseModel):
    """One sliding window of raw sensor readings submitted for diagnosis."""

    asset_id: str
    timestamps: list[str] = Field(..., description="ISO 8601 timestamps, one per sample")
    readings: dict[str, list[float]] = Field(..., description="signal name -> list of samples")

    @field_validator("readings")
    @classmethod
    def _check_signals(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        missing = [s for s in SENSOR_NAMES if s not in v]
        if missing:
            raise ValueError(f"missing required signals: {missing}")
        return v

    def sample_count(self) -> int:
        return len(self.timestamps)


class DataQuality(BaseModel):
    missing_value_count: int = 0
    out_of_range_count: int = 0
    sample_count: int = 0
    sufficient: bool = True
    notes: list[str] = Field(default_factory=list)


class TopFeature(BaseModel):
    feature: str
    direction: Literal["high", "low", "normal"]
    weight: float


class DiagnosticRecord(BaseModel):
    """Misan -> physics validation / optimiser / dashboard output contract (PRD Section 8)."""

    record_id: str
    timestamp: str
    asset_id: str
    fault_detected: bool
    fault_class: str
    component: str
    group: str
    class_probabilities: dict[str, float]
    confidence: float
    severity_index: float | None = None
    severity_level: SeverityLevel | None = None
    severity_trend: SeverityTrend = "unknown"
    top_features: list[TopFeature] = Field(default_factory=list)
    explanation: str
    validation_status: ValidationStatus
    validation_note: str
    model_version: str
    data_quality: DataQuality | None = None


class EvaluationSummary(BaseModel):
    """Benchmark report produced by misan.evaluation (PRD Section 11 / FR-32)."""

    model_version: str
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    false_alarm_rate: float
    detection_delay_windows: float | None
    detection_miss_rate: float
    detection_run_recall: float
    n_faulty_runs: int
    n_detected_faulty_runs: int
    calibration_error: float
    inference_time_ms: float
    noise_robustness: dict[str, float]
    baseline_macro_f1: float
    baseline_false_alarm_rate: float
    physics_confirmation_rate: float
    physics_rejection_rate: float
    n_test_windows: int
    generated_at: str
