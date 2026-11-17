"""Entraînement chronologique, comparaison à une baseline et prévision."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from clearair import config as C
from clearair.anomaly import fit as fit_anomaly
from clearair.features import build_features


def _regressor(loss: str, alpha: float | None = None) -> HistGradientBoostingRegressor:
    options: dict[str, Any] = {
        "loss": loss,
        "max_iter": 100,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "early_stopping": True,
        "random_state": 42,
    }
    if alpha is not None:
        options["quantile"] = alpha
    return HistGradientBoostingRegressor(**options)


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _top_drivers(estimator, X_test, y_test, columns, top: int = 8) -> list[dict[str, Any]]:
    """Importance par permutation (MAE) sur un sous-échantillon : ce qui guide la prévision."""
    size = min(1500, len(y_test))
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(y_test), size=size, replace=False))
    result = permutation_importance(
        estimator, X_test[pick], y_test.to_numpy()[pick], n_repeats=3, random_state=0,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)[::-1][:top]
    return [{"feature": columns[i], "mae_increase": float(result.importances_mean[i])}
            for i in order]


def train_all(
    data: pd.DataFrame,
    model_path: str | Path = C.FORECAST_MODEL_PATH,
    metrics_path: str | Path = C.METRICS_PATH,
    anomaly_path: str | Path = C.ANOMALY_MODEL_PATH,
    city: str = C.CITY,
) -> dict[str, Any]:
    """Entraîne un modèle par horizon et évalue sur la fin de la série."""
    frame = data.sort_index().copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Les données d'entraînement doivent avoir un index datetime.")
    if "pm10" not in frame or frame["pm10"].notna().sum() < 150:
        raise ValueError("Il faut au moins 150 mesures PM10 valides pour entraîner.")
    X_all = build_features(frame)
    bundle: dict[str, Any] = {
        "target": "pm10",
        "horizons": {},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    report: dict[str, Any] = {
        "city": city,
        "trained_at": bundle["trained_at"],
        "rows": int(len(frame)),
        "target": "pm10",
        "horizons": {},
    }

    for horizon in C.HORIZONS:
        target = frame["pm10"].shift(-horizon).rename("target")
        dataset = X_all.join(target).dropna(subset=["pm10", "target"])
        if len(dataset) < 100:
            raise ValueError(
                f"Pas assez de lignes utilisables pour l'horizon {horizon} h "
                f"({len(dataset)} ; 100 minimum)."
            )
        n = len(dataset)
        test_at = int(n * (1 - C.TEST_FRACTION))
        calib_at = int(test_at * (1 - C.CALIB_FRACTION))
        if calib_at < 50 or test_at - calib_at < 20 or n - test_at < 20:
            raise ValueError(f"Série trop courte pour évaluer l'horizon {horizon} h.")
        train = dataset.iloc[:calib_at]
        calib = dataset.iloc[calib_at:test_at]
        test = dataset.iloc[test_at:]
        feature_columns = [
            column for column in X_all.columns if train[column].notna().any()
        ]
        if not feature_columns:
            raise ValueError(f"Aucune variable exploitable pour l'horizon {horizon} h.")
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train[feature_columns])
        X_calib = imputer.transform(calib[feature_columns])
        X_test = imputer.transform(test[feature_columns])
        y_train, y_calib, y_test = train["target"], calib["target"], test["target"]

        estimators: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        calib_predictions: dict[str, np.ndarray] = {}
        for label, loss, alpha in (
            ("lower", "quantile", C.QUANTILES[0]),
            ("median", "squared_error", None),
            ("upper", "quantile", C.QUANTILES[2]),
        ):
            estimator = _regressor(loss, alpha)
            estimator.fit(X_train, y_train)
            estimators[label] = estimator
            predictions[label] = estimator.predict(X_test)
            calib_predictions[label] = estimator.predict(X_calib)

        # Calibration conformelle des quantiles (CQR) : corrige l'intervalle brut,
        # souvent trop étroit, avec les erreurs observées sur un bloc de calibration.
        lo_c = np.minimum(calib_predictions["lower"], calib_predictions["upper"])
        hi_c = np.maximum(calib_predictions["lower"], calib_predictions["upper"])
        scores = np.maximum(lo_c - y_calib.to_numpy(), y_calib.to_numpy() - hi_c)
        target_cov = C.QUANTILES[2] - C.QUANTILES[0]
        level = min(1.0, np.ceil((len(scores) + 1) * target_cov) / len(scores))
        conformal_q = float(np.quantile(scores, level))
        lo_t = np.minimum(predictions["lower"], predictions["upper"])
        hi_t = np.maximum(predictions["lower"], predictions["upper"])
        cov_raw = float(np.mean((y_test.to_numpy() >= lo_t) & (y_test.to_numpy() <= hi_t)))
        cov_cal = float(np.mean((y_test.to_numpy() >= lo_t - conformal_q)
                                & (y_test.to_numpy() <= hi_t + conformal_q)))

        # Probabilité de dépasser le repère exploratoire (100 µg/m³) à cet horizon.
        exceed_train = (y_train >= C.EXPLORATORY_PM10).astype(int)
        classifier = None
        exceed_report: dict[str, Any] = {"available": False}
        if exceed_train.nunique() == 2 and exceed_train.value_counts().min() >= 30:
            classifier = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.08, max_leaf_nodes=15,
                min_samples_leaf=20, early_stopping=True, random_state=42,
            ).fit(X_train, exceed_train)
            y_exceed = (y_test >= C.EXPLORATORY_PM10).astype(int)
            proba = classifier.predict_proba(X_test)[:, 1]
            base_rate = float(exceed_train.mean())
            exceed_report = {
                "available": True,
                "base_rate_train": base_rate,
                "brier_model": float(brier_score_loss(y_exceed, proba)),
                "brier_climatology": float(np.mean((y_exceed - base_rate) ** 2)),
                "brier_persistence": float(np.mean(
                    (y_exceed - (test["pm10"].to_numpy() >= C.EXPLORATORY_PM10)) ** 2)),
                "roc_auc": (float(roc_auc_score(y_exceed, proba))
                            if y_exceed.nunique() == 2 else None),
            }

        baseline = test["pm10"].to_numpy()
        median_prediction = predictions["median"]
        model_m, base_m = _metrics(y_test, median_prediction), _metrics(y_test, baseline)
        in_season = test.index.month.isin([11, 12, 1, 2, 3])
        by_season = {}
        for name, mask in (("harmattan", in_season), ("hors_harmattan", ~in_season)):
            if mask.sum() >= 20:
                by_season[name] = {
                    "rows": int(mask.sum()),
                    "baseline_mae": float(mean_absolute_error(y_test[mask], baseline[mask])),
                    "model_mae": float(mean_absolute_error(y_test[mask], median_prediction[mask])),
                }
        report["horizons"][str(horizon)] = {
            "train_rows": int(len(train)),
            "calibration_rows": int(len(calib)),
            "test_rows": int(len(test)),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "baseline": base_m,
            "model": model_m,
            "skill_vs_persistence": float(1 - model_m["mae"] / base_m["mae"]),
            "coverage_80_raw": cov_raw,
            "coverage_80": cov_cal,
            "conformal_width_added": conformal_q,
            "by_season": by_season,
            "exceedance_100": exceed_report,
        }
        bundle["horizons"][str(horizon)] = {
            "feature_columns": feature_columns,
            "imputer": imputer,
            "estimators": estimators,
            "conformal_q": conformal_q,
            "classifier": classifier,
        }
        if horizon == max(C.HORIZONS):
            report["drivers_24h"] = _top_drivers(estimators["median"], X_test, y_test, feature_columns)

    anomaly_file = Path(anomaly_path)
    anomaly_file.parent.mkdir(parents=True, exist_ok=True)
    anomaly_train_end = int(len(frame) * (1 - C.TEST_FRACTION))
    anomaly_model = fit_anomaly(frame.iloc[:anomaly_train_end])
    joblib.dump(anomaly_model, anomaly_file)
    report["anomaly"] = {
        "model": "IsolationForest",
        "training_rows": int(anomaly_train_end),
        "interpretation": "Alerte statistique exploratoire, non diagnostic de poussière.",
    }

    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file)
    metrics_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_forecaster(path: str | Path = C.FORECAST_MODEL_PATH) -> dict[str, Any]:
    """Charge un bundle de prévision existant ou signale son absence."""
    model_file = Path(path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Modèle absent : {model_file}. Lancez python train.py.")
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or "horizons" not in bundle:
        raise ValueError(f"Format de modèle non reconnu : {model_file}")
    return bundle


def predict(bundle: dict[str, Any], features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Prédit médiane et intervalle empirique par horizon."""
    output = {}
    for horizon, model in bundle["horizons"].items():
        required = model.get("feature_columns")
        estimators = model.get("estimators")
        imputer = model.get("imputer")
        if not isinstance(required, list) or not required or not isinstance(estimators, dict):
            raise ValueError(f"Format de modèle non reconnu pour l'horizon {horizon} h.")
        absent = set(required) - set(features.columns)
        if absent:
            raise ValueError(f"Variables manquantes pour la prévision : {sorted(absent)}")
        row = features.iloc[[-1]][required]
        row = imputer.transform(row)
        lower = float(estimators["lower"].predict(row)[0])
        median = float(estimators["median"].predict(row)[0])
        upper = float(estimators["upper"].predict(row)[0])
        margin = float(model.get("conformal_q", 0.0))
        classifier = model.get("classifier")
        output[horizon] = {
            "lower": max(0.0, min(lower, upper) - margin),
            "median": max(0.0, median),
            "upper": max(0.0, max(lower, upper) + margin),
            "p_exceed": (float(classifier.predict_proba(row)[0, 1])
                         if classifier is not None else None),
        }
    return output
