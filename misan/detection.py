"""Abnormality detection and the alarm persistence rule (PRD FR-7, FR-8, FR-9).

Detection is trained as a binary healthy/faulty classifier, kept independent from the
multiclass diagnosis model so the two concerns (``is something wrong`` vs ``what is
wrong``) can be evaluated and improved separately, per PRD Section 7.2.
"""
from __future__ import annotations

from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from misan.features import FEATURE_COLUMNS


class DetectionModel:
    """Binary healthy (0) vs faulty (1) classifier over the shared feature table."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced", n_jobs=-1
        )
        self.feature_columns: list[str] = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y_fault_class: pd.Series) -> None:
        y_binary = (y_fault_class != "NORMAL").astype(int)
        self.model.fit(X[self.feature_columns], y_binary)

    def predict_is_faulty(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_columns])

    def predict_proba_faulty(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DetectionModel":
        return joblib.load(path)


class PersistenceTracker:
    """Requires N consecutive abnormal windows per asset before raising an alert (FR-9).

    State is kept in memory per running process. This is a deliberate MVP simplification
    (documented in the status report): the counter resets on service restart, which only
    delays — never falsely raises — an alert.
    """

    def __init__(self, n: int):
        self.n = n
        self._history: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=self.n))

    def update(self, asset_id: str, is_abnormal: bool) -> bool:
        history = self._history[asset_id]
        history.append(is_abnormal)
        return len(history) == self.n and all(history)

    def reset(self, asset_id: str | None = None) -> None:
        if asset_id is None:
            self._history.clear()
        else:
            self._history.pop(asset_id, None)
