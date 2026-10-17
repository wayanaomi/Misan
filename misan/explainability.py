"""SHAP-based explainability (PRD FR-16, FR-17, FR-18).

Uses the diagnosis model's uncalibrated explainer clone so `shap.TreeExplainer` can be
used directly against tree-based estimators for fast, exact attributions. Falls back to
a generic permutation explainer for non-tree comparison models (SVC, KNN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from misan.diagnosis import DiagnosisModel
from misan.schemas import TopFeature

_TREE_MODEL_TYPES = ("RandomForest", "GradientBoosting", "ExtraTrees", "XGB")


def _is_tree_model(estimator) -> bool:
    return any(t in type(estimator).__name__ for t in _TREE_MODEL_TYPES)


def explain(diagnosis_model: DiagnosisModel, X_row: pd.DataFrame, predicted_class: str, top_k: int = 5) -> list[TopFeature]:
    """Return the top_k contributing features for `predicted_class` on a single window."""
    estimator = diagnosis_model.explainer_estimator
    Xf = X_row[diagnosis_model.feature_columns]

    if _is_tree_model(estimator):
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(Xf)
        class_index = list(estimator.classes_).index(predicted_class) if predicted_class in list(estimator.classes_) else 0
        values = raw[class_index][0] if isinstance(raw, list) else np.asarray(raw)[0, :, class_index]
    else:
        background = diagnosis_model.training_mean.to_frame().T if diagnosis_model.training_mean is not None else Xf
        explainer = shap.Explainer(estimator.predict_proba, background, algorithm="permutation")
        raw = explainer(Xf, max_evals=200)
        class_index = list(estimator.classes_).index(predicted_class) if predicted_class in list(estimator.classes_) else 0
        values = raw.values[0, :, class_index]

    order = np.argsort(-np.abs(values))[:top_k]
    features: list[TopFeature] = []
    for idx in order:
        name = diagnosis_model.feature_columns[idx]
        shap_value = float(values[idx])
        observed = float(Xf.iloc[0, idx])
        baseline = float(diagnosis_model.training_mean[name]) if diagnosis_model.training_mean is not None else observed
        direction = "high" if observed >= baseline else "low"
        features.append(TopFeature(feature=name, direction=direction, weight=round(shap_value, 4)))
    return features


def to_sentence(top_features: list[TopFeature], max_features: int = 3) -> str:
    """Translate the leading SHAP evidence into one plain-language sentence (FR-17)."""
    if not top_features:
        return "No contributing features could be attributed to this window."
    readable = []
    for f in top_features[:max_features]:
        label = f.feature.replace("_mean", "").replace("_slope", " trend").replace("_std", " variability").replace("_range", " swing")
        label = label.replace("_", " ")
        readable.append(f"{label} is {f.direction}")
    if len(readable) == 1:
        body = readable[0]
    elif len(readable) == 2:
        body = " and ".join(readable)
    else:
        body = ", ".join(readable[:-1]) + f", and {readable[-1]}"
    return body[0].upper() + body[1:] + "."
