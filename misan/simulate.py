"""Physics-based synthetic data simulator — a DEVELOPMENT/RESEARCH TOOL ONLY.

This generates labelled training/evaluation data for Misan using a simplified,
explicitly documented steady-state vapour-compression cycle model. It intentionally
reproduces the expected sensor-direction behaviour encoded in `config/signatures.yaml`
so the physics-signature check and the learned models are evaluated against physically
consistent fault patterns.

It is invoked only through an explicit command (`python -m misan.cli simulate`) and
never runs automatically or writes into the operational database — see PRD Section 3,
"Zero demo data". Output files live under data/raw/ and must be explicitly imported
or pointed at for training; they are not loaded into the live application.

Important limitation (documented, not hidden): this is a simplified proxy model for
MVP demonstration, not the project's full physics-based simulator. Results trained on
it describe how well Misan's pipeline works on this proxy, not on a real plant.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from misan.config import SAMPLING_INTERVAL_SECONDS

REFERENCE_AMBIENT_C = 35.0
REFERENCE_LOAD_SETPOINT_C = 18.0

FAULT_SEVERITIES = (0.25, 0.5, 0.75, 1.0)
NOISE_LEVELS = (0.0, 0.01, 0.03, 0.06)  # fraction of signal std added as Gaussian noise


@dataclass
class OperatingCondition:
    ambient_temperature: float
    load_setpoint: float


def _healthy_baseline(n: int, cond: OperatingCondition, rng: np.random.Generator) -> pd.DataFrame:
    ambient_shift = cond.ambient_temperature - REFERENCE_AMBIENT_C
    load_shift = cond.load_setpoint - REFERENCE_LOAD_SETPOINT_C

    drift = rng.normal(0, 1, n).cumsum() * 0.02  # slow random walk so windows aren't perfectly flat
    t = np.arange(n)

    suction_pressure = 900 + 0.3 * load_shift - 1.0 * ambient_shift + drift
    discharge_pressure = 2200 + 15 * ambient_shift + 5 * load_shift + drift * 2
    suction_temperature = 5 + 0.1 * load_shift + drift * 0.3
    discharge_temperature = 70 + 0.6 * ambient_shift + drift * 0.4
    condenser_inlet_temperature = np.full(n, cond.ambient_temperature) + drift * 0.1
    condenser_outlet_temperature = condenser_inlet_temperature + 10 + 0.1 * ambient_shift
    evaporator_inlet_temperature = np.full(n, cond.load_setpoint) + 2 + drift * 0.1
    evaporator_outlet_temperature = evaporator_inlet_temperature - 8
    refrigerant_mass_flow = 400 + 3 * load_shift - 1 * ambient_shift + drift * 2
    compressor_power = 5 + 0.05 * load_shift + 0.08 * ambient_shift + drift * 0.05
    ambient_temperature = np.full(n, cond.ambient_temperature)
    load_setpoint = np.full(n, cond.load_setpoint)

    return pd.DataFrame(
        {
            "t": t,
            "suction_pressure": suction_pressure,
            "discharge_pressure": discharge_pressure,
            "suction_temperature": suction_temperature,
            "discharge_temperature": discharge_temperature,
            "condenser_inlet_temperature": condenser_inlet_temperature,
            "condenser_outlet_temperature": condenser_outlet_temperature,
            "evaporator_inlet_temperature": evaporator_inlet_temperature,
            "evaporator_outlet_temperature": evaporator_outlet_temperature,
            "refrigerant_mass_flow": refrigerant_mass_flow,
            "compressor_power": compressor_power,
            "ambient_temperature": ambient_temperature,
            "load_setpoint": load_setpoint,
        }
    )


def _apply_fault(df: pd.DataFrame, fault_class: str, severity: float) -> pd.DataFrame:
    df = df.copy()
    s = severity
    n = len(df)
    ramp = np.linspace(0, 1, n)  # faults ramp in rather than snapping on, more realistic

    if fault_class == "NORMAL":
        return df

    if fault_class == "CMP-DEG":
        df["discharge_pressure"] += 220 * s * ramp
        df["compressor_power"] += 1.8 * s * ramp
        df["refrigerant_mass_flow"] -= 40 * s * ramp

    elif fault_class == "EXV-STICK":
        df["suction_temperature"] += 7 * s * ramp
        df["evaporator_outlet_temperature"] += 6 * s * ramp
        df["refrigerant_mass_flow"] -= 70 * s * ramp
        df["suction_pressure"] -= 60 * s * ramp

    elif fault_class == "CND-FOUL":
        df["discharge_pressure"] += 260 * s * ramp
        df["condenser_outlet_temperature"] += 4 * s * ramp
        df["compressor_power"] += 1.4 * s * ramp

    elif fault_class == "EVP-FOUL":
        df["suction_pressure"] -= 90 * s * ramp
        df["suction_temperature"] += 4 * s * ramp
        df["refrigerant_mass_flow"] -= 30 * s * ramp

    elif fault_class == "REF-LEAK":
        df["suction_pressure"] -= 140 * s * ramp
        df["discharge_pressure"] -= 220 * s * ramp
        df["suction_temperature"] += 8 * s * ramp
        df["refrigerant_mass_flow"] -= 110 * s * ramp

    elif fault_class == "SNS-DRIFT":
        df["suction_temperature"] += 10 * s * ramp  # sensor reading drifts; the physical state is unchanged

    elif fault_class == "SP-ERR":
        df["load_setpoint"] += 6 * s * ramp
        df["evaporator_inlet_temperature"] += 5 * s * ramp
        df["evaporator_outlet_temperature"] += 3 * s * ramp

    else:
        raise ValueError(f"unknown fault class: {fault_class}")

    return df


def simulate_run(
    fault_class: str,
    severity: float,
    cond: OperatingCondition,
    duration_minutes: int,
    noise_level: float,
    rng: np.random.Generator,
    run_id: str,
    asset_id: str = "SIM-ASSET-1",
) -> pd.DataFrame:
    n = duration_minutes
    df = _healthy_baseline(n, cond, rng)
    df = _apply_fault(df, fault_class, severity)

    signal_cols = [c for c in df.columns if c != "t"]
    if noise_level > 0:
        for col in signal_cols:
            sigma = max(abs(df[col].std()), 1e-3) * noise_level
            df[col] = df[col] + rng.normal(0, sigma, n)

    start = pd.Timestamp("2026-01-01T00:00:00")
    df["timestamp"] = [start + pd.Timedelta(seconds=SAMPLING_INTERVAL_SECONDS * i) for i in range(n)]
    df["run_id"] = run_id
    df["asset_id"] = asset_id
    df["fault_class"] = fault_class
    df["severity"] = severity if fault_class != "NORMAL" else 0.0
    df["noise_level"] = noise_level
    return df.drop(columns=["t"])


def generate_dataset(
    fault_classes: list[str],
    runs_per_condition: int = 3,
    duration_minutes: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a full labelled dataset spanning all fault classes, several severities,
    several operating conditions and several noise levels (PRD FR-28 to FR-30, 9.2)."""
    rng = np.random.default_rng(seed)
    conditions = [
        OperatingCondition(ambient_temperature=amb, load_setpoint=load)
        for amb in (25.0, 35.0, 45.0)
        for load in (16.0, 18.0, 22.0)
    ]
    runs: list[pd.DataFrame] = []
    for fault_class in fault_classes:
        severities = (0.0,) if fault_class == "NORMAL" else FAULT_SEVERITIES
        for severity in severities:
            for cond in conditions:
                for _ in range(runs_per_condition):
                    noise_level = rng.choice(NOISE_LEVELS)
                    run_id = f"{fault_class}-{uuid.uuid4().hex[:8]}"
                    runs.append(
                        simulate_run(
                            fault_class=fault_class,
                            severity=severity,
                            cond=cond,
                            duration_minutes=duration_minutes,
                            noise_level=float(noise_level),
                            rng=rng,
                            run_id=run_id,
                        )
                    )
    return pd.concat(runs, ignore_index=True)
