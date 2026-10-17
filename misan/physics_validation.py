"""Lightweight physics-based validation: fault-signature consistency check
(PRD FR-19, FR-20, FR-21, FR-23).

This is a placeholder for the real physics-validation service (FR-22) — a signature
table encodes the expected direction of change for each fault class relative to a
healthy baseline, and a diagnosis is confirmed/rejected by majority vote across its
signature rules. Classes with no signature (e.g. NORMAL) or without a usable baseline
are reported `unchecked`, never silently treated as confirmed.
"""
from __future__ import annotations

import pandas as pd

from misan.config import SIGNATURES, SignatureRule


def check_signature(
    feature_row: pd.Series,
    fault_class: str,
    baseline_mean: pd.Series | None,
    tolerance_std: pd.Series | None = None,
    min_margin: float = 0.05,
) -> tuple[str, str]:
    """Returns (validation_status, validation_note)."""
    rules: list[SignatureRule] = SIGNATURES.get(fault_class, [])
    if not rules:
        return "unchecked", f"no physics signature defined for {fault_class}"
    if baseline_mean is None:
        return "unchecked", "no healthy baseline available to compare against"

    agree, disagree, skipped = 0, 0, 0
    for rule in rules:
        if rule.feature not in feature_row.index or rule.feature not in baseline_mean.index:
            skipped += 1
            continue
        observed = feature_row[rule.feature]
        base = baseline_mean[rule.feature]
        spread = abs(base) * min_margin if tolerance_std is None else float(tolerance_std.get(rule.feature, 0.0))
        margin = max(spread, 1e-6)
        if rule.direction == "high":
            matches = observed > base + margin
        else:
            matches = observed < base - margin
        if matches:
            agree += 1
        else:
            disagree += 1

    total = agree + disagree
    if total == 0:
        return "unchecked", "signature features were not available in this window"
    if agree >= total - agree:  # majority (ties favour confirmation, matching a lenient preliminary check)
        return "confirmed", f"{agree}/{total} expected sensor-direction signatures matched"
    return "rejected", f"only {agree}/{total} expected sensor-direction signatures matched"


def resolve_with_fallback(
    class_probabilities: dict[str, float],
    baseline_mean: pd.Series | None,
    feature_row: pd.Series,
    top_class: str,
    min_probability: float = 0.01,
) -> tuple[str, str, str, bool]:
    """FR-23: if the top class is rejected, try the next most probable class that
    passes the check. Candidates with near-zero model probability are skipped — a
    class the model assigns effectively no weight to is not "the next most probable
    class", even if its signature happens to pass. Returns
    (final_class, status, note, fallback_used)."""
    status, note = check_signature(feature_row, top_class, baseline_mean)
    if status != "rejected":
        return top_class, status, note, False

    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for code, probability in ranked:
        if code == top_class or probability < min_probability:
            continue
        alt_status, alt_note = check_signature(feature_row, code, baseline_mean)
        if alt_status == "confirmed":
            return code, alt_status, f"fallback from {top_class} ({note}); {alt_note}", True
    return top_class, status, note, False
