"""Fault diagnosis: multiclass classification with probability calibration
(PRD FR-10, FR-11, FR-12, FR-14)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from misan.config import FAULT_CODES
from misan.features import FEATURE_COLUMNS


class DiagnosisModel:
    """Wraps a calibrated multiclass classifier. `estimator` is any scikit-learn
    classifier (RandomForest, GradientBoosting, SVC, KNN, ...) selected during training
    by comparing validation performance (PRD Section 10)."""

    def __init__(self, estimator, calibrate: bool = True, cv: int = 3):
        self.base_estimator = estimator
        self.calibrate = calibrate
        self.cv = cv
        self.model = None
        self.explainer_estimator = None  # uncalibrated clone, used only for SHAP (PRD FR-16)
        self.training_mean: pd.Series | None = None
        self.feature_columns: list[str] = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = X[self.feature_columns]
        if self.calibrate:
            self.model = CalibratedClassifierCV(self.base_estimator, method="isotonic", cv=self.cv)
        else:
            self.model = self.base_estimator
        self.model.fit(Xf, y)
        self.explainer_estimator = clone(self.base_estimator)
        self.explainer_estimator.fit(Xf, y)
        self.training_mean = Xf.mean()

    def predict_proba_table(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of shape (n, len(FAULT_CODES)) — zero-filled for classes
        absent from the training data, so every row sums close to 1 over FAULT_CODES."""
        Xf = X[self.feature_columns]
        proba = self.model.predict_proba(Xf)
        classes = list(self.model.classes_)
        table = pd.DataFrame(proba, columns=classes, index=X.index)
        for code in FAULT_CODES:
            if code not in table.columns:
                table[code] = 0.0
        return table[FAULT_CODES]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba_table = self.predict_proba_table(X)
        return proba_table.idxmax(axis=1).to_numpy()

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DiagnosisModel":
        return joblib.load(path)


def class_probabilities(row: pd.Series) -> dict[str, float]:
    return {code: float(row[code]) for code in FAULT_CODES}


def top_class(row: pd.Series) -> tuple[str, float]:
    code = row.idxmax()
    return str(code), float(row[code])
