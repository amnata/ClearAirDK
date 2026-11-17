"""Construction des variables prédictives sans utiliser d'observations futures."""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 3, 6, 12, 24, 48, 72)
ROLLING_WINDOWS = (6, 24, 72)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construit les retards, moyennes passées et variables calendaires."""
    frame = data.copy().sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Les données doivent être indexées par un DatetimeIndex.")
    features = pd.DataFrame(index=frame.index)
    for column in ("pm10", "pm2_5", "dust", "wind_speed", "wind_dir", "rh", "temp"):
        if column not in frame.columns:
            frame[column] = np.nan
        features[column] = pd.to_numeric(frame[column], errors="coerce")
        if column in ("pm10", "pm2_5", "dust"):
            for lag in LAGS:
                features[f"{column}_lag{lag}"] = features[column].shift(lag)
            for window in ROLLING_WINDOWS:
                features[f"{column}_mean{window}"] = (
                    features[column].shift(1).rolling(window, min_periods=window).mean()
                )
    idx = features.index
    features["hour"] = idx.hour
    features["day_of_year_sin"] = np.sin(2 * np.pi * idx.day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * idx.day_of_year / 365.25)
    features["is_harmattan_season"] = idx.month.isin([11, 12, 1, 2, 3]).astype(int)
    return features.replace([np.inf, -np.inf], np.nan)
