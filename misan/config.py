"""Central configuration: environment settings plus the YAML-driven fault taxonomy,
sensor schema and physics signature table (PRD Section 12, Maintainability)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

# Support both the source checkout and an installed package on Railway.
_SOURCE_CONFIG_DIR = ROOT_DIR / "config"
_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent / "config"

if (_SOURCE_CONFIG_DIR / "faults.yaml").is_file():
    CONFIG_DIR = _SOURCE_CONFIG_DIR
else:
    CONFIG_DIR = _PACKAGED_CONFIG_DIR


def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: ROOT_DIR / _getenv("MISAN_DB_PATH", "data/misan.db"))
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / _getenv("MISAN_DATA_DIR", "data"))
    models_dir: Path = field(default_factory=lambda: ROOT_DIR / _getenv("MISAN_MODELS_DIR", "data/models"))
    api_base_url: str = field(default_factory=lambda: _getenv("MISAN_API_BASE_URL", "http://127.0.0.1:8000"))
    persistence_n: int = field(default_factory=lambda: int(_getenv("MISAN_PERSISTENCE_N", "3")))
    confidence_threshold: float = field(default_factory=lambda: float(_getenv("MISAN_CONFIDENCE_THRESHOLD", "0.45")))


SETTINGS = Settings()
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.models_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FaultClass:
    code: str
    name: str
    component: str
    group: str


@dataclass(frozen=True)
class SensorSignal:
    name: str
    unit: str
    valid_range: tuple[float, float]


@dataclass(frozen=True)
class SignatureRule:
    feature: str
    direction: Literal["high", "low"]


def load_fault_classes() -> list[FaultClass]:
    raw = yaml.safe_load((CONFIG_DIR / "faults.yaml").read_text())
    return [FaultClass(**c) for c in raw["classes"]]


def load_sensor_schema() -> tuple[int, int, list[SensorSignal]]:
    raw = yaml.safe_load((CONFIG_DIR / "sensors.yaml").read_text())
    signals = [SensorSignal(name=s["name"], unit=s["unit"], valid_range=tuple(s["valid_range"])) for s in raw["signals"]]
    return raw["sampling_interval_seconds"], raw["window_length_samples"], signals


def load_signatures() -> dict[str, list[SignatureRule]]:
    raw = yaml.safe_load((CONFIG_DIR / "signatures.yaml").read_text())
    return {code: [SignatureRule(**r) for r in rules] for code, rules in raw.items()}


FAULT_CLASSES: list[FaultClass] = load_fault_classes()
FAULT_BY_CODE: dict[str, FaultClass] = {f.code: f for f in FAULT_CLASSES}
SAMPLING_INTERVAL_SECONDS, WINDOW_LENGTH_SAMPLES, SENSOR_SIGNALS = load_sensor_schema()
SENSOR_NAMES: list[str] = [s.name for s in SENSOR_SIGNALS]
SENSOR_BY_NAME: dict[str, SensorSignal] = {s.name: s for s in SENSOR_SIGNALS}
SIGNATURES: dict[str, list[SignatureRule]] = load_signatures()
FAULT_CODES: list[str] = [f.code for f in FAULT_CLASSES]
