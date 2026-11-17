"""Détection exploratoire des observations atypiques."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

FEATURE_COLUMNS = ["pm10", "pm2_5", "dust", "wind_speed"]


def fit(data: pd.DataFrame) -> object:
    """Entraîne un détecteur exploratoire Isolation Forest."""
    columns = data.reindex(columns=FEATURE_COLUMNS)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        IsolationForest(contamination=0.02, random_state=42, n_estimators=150),
    )
    model.fit(columns)
    return model


def score(model: object, data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute un indicateur statistique d'anomalie aux lignes fournies."""
    columns = data.reindex(columns=FEATURE_COLUMNS)
    result = data.copy()
    result["is_unusual"] = model.predict(columns) == -1
    result["anomaly_score"] = model.decision_function(columns)
    return result
