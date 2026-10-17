"""Rule-based FDD baseline for comparison against Misan's learned models (PRD FR-31).

Raises an alarm only when a monitored feature crosses a z-score threshold relative to
the healthy baseline, then attributes the alarm to whichever fault signature (see
`misan.physics_validation`) has the most matching expected-direction rules. This is
intentionally simple — it is the "simple rule-based method" Misan must beat, not a
second learned model.
"""
from __future__ import annotations

import pandas as pd

from misan.config import SIGNATURES


class RuleBasedBaseline:
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self.baseline_mean: pd.Series | None = None
        self.baseline_std: pd.Series | None = None

    def fit(self, X_healthy: pd.DataFrame) -> None:
        self.baseline_mean = X_healthy.mean()
        self.baseline_std = X_healthy.std().replace(0, 1e-6)

    def _z_scores(self, row: pd.Series) -> pd.Series:
        return (row - self.baseline_mean) / self.baseline_std

    def predict_one(self, row: pd.Series) -> str:
        if self.baseline_mean is None:
            raise RuntimeError("baseline has not been fit")
        z = self._z_scores(row)
        any_alarm = (z.abs() > self.z_threshold).any()
        if not any_alarm:
            return "NORMAL"

        best_class, best_score = "NORMAL", 0
        for code, rules in SIGNATURES.items():
            if not rules:
                continue
            score = 0
            for rule in rules:
                if rule.feature not in z.index:
                    continue
                value = z[rule.feature]
                if rule.direction == "high" and value > self.z_threshold:
                    score += 1
                elif rule.direction == "low" and value < -self.z_threshold:
                    score += 1
            if score > best_score:
                best_class, best_score = code, score
        return best_class

    def predict(self, X: pd.DataFrame) -> list[str]:
        return [self.predict_one(X.iloc[i]) for i in range(len(X))]
