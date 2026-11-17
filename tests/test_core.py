import numpy as np
import pandas as pd
import pytest

from clearair import advice
from clearair import anomaly
from clearair.data import clean
from clearair.features import build_features
from clearair.insights import (
    best_outdoor_window,
    detect_particle_episodes,
    monthly_summary,
    split_observed_forecast,
)
from clearair.models import train_all, predict, load_forecaster


def synthetic(n_days: int = 45, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    seasonal = np.where(index.month.isin([11, 12, 1, 2, 3]), 1.8, 1.0)
    pm10 = np.maximum(5, 25 * seasonal + 12 * np.sin(2 * np.pi * index.hour / 24)
                      + rng.gamma(2, 5, len(index)))
    frame = pd.DataFrame({
        "pm10": pm10,
        "pm2_5": pm10 * 0.35,
        "dust": pm10 * 0.6,
        "temp": 27 + rng.normal(0, 2, len(index)),
        "rh": 55 + rng.normal(0, 8, len(index)),
        "wind_speed": rng.uniform(1, 8, len(index)),
        "wind_dir": rng.uniform(0, 360, len(index)),
    }, index=index)
    frame.index.name = "time"
    return frame


def test_levels_and_missing_data():
    assert advice.level_from(10) == 0
    assert advice.level_from(300) == 4
    assert advice.level_from(20, 55) == 3
    assert advice.level_from(10, 30) == 2
    assert advice.level_from(None, None) is None


def test_advice_exists_for_every_profile_and_level():
    for profile in advice.PROFILES:
        for level in range(5):
            assert advice.advice_for(profile, level)
    assert "insuffisantes" in advice.advice_for("enfant", None)
    with pytest.raises(ValueError):
        advice.advice_for("unknown", 1)


def test_clean_parses_timestamp_removes_duplicates_and_negative_values():
    result = clean(pd.DataFrame({
        "time": ["2024-01-01 00:00", "2024-01-01 00:00", "bad"],
        "pm10": [10, 20, 15],
    }))
    assert len(result) == 1
    assert result.iloc[0]["pm10"] == 20
    assert pd.isna(clean(pd.DataFrame({"time": ["2024-01-01"], "pm10": [-1]}))
                   .iloc[0]["pm10"])


def test_features_use_only_present_and_past_observations():
    frame = synthetic()
    features = build_features(frame)
    position = 100
    assert features.iloc[position]["pm10_lag24"] == frame["pm10"].iloc[position - 24]
    assert features.iloc[position]["pm10"] == frame["pm10"].iloc[position]
    assert features.iloc[position]["pm10_mean24"] == pytest.approx(
        frame["pm10"].iloc[position - 24:position].mean()
    )


def test_training_evaluation_and_inference(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    anomaly_path = tmp_path / "anomaly.joblib"
    frame = synthetic(60).drop(columns=["temp", "rh", "wind_speed", "wind_dir"])
    report = train_all(frame, model_path=model_path, metrics_path=metrics_path,
                       anomaly_path=anomaly_path)
    assert set(report["horizons"]) == {"6", "12", "24"}
    assert anomaly_path.is_file()
    assert report["anomaly"]["model"] == "IsolationForest"
    for metrics in report["horizons"].values():
        assert 0 <= metrics["coverage_80"] <= 1
        assert metrics["test_rows"] > 0
        assert "mae" in metrics["baseline"] and "mae" in metrics["model"]
    bundle = load_forecaster(model_path)
    result = predict(bundle, build_features(frame))
    assert set(result) == {"6", "12", "24"}
    assert all(value["lower"] <= value["upper"] for value in result.values())


def test_anomaly_detector_flags_extreme_particle_spike():
    frame = synthetic(35)
    detector = anomaly.fit(frame)
    spike = frame.iloc[[400]].copy()
    spike.loc[:, ["pm10", "pm2_5", "dust"]] = [3000, 1200, 2800]
    assert bool(anomaly.score(detector, spike)["is_unusual"].iloc[0])


def test_particle_episodes_require_consecutive_hourly_observations():
    index = pd.date_range("2024-01-01", periods=11, freq="h")
    frame = pd.DataFrame({
        "pm10": [20, 110, 120, 105, 20, 110, 110, np.nan, 120, 130, 140],
        "dust": [10, 40, 50, 45, 10, 30, 30, np.nan, 60, 70, 80],
    }, index=index)
    episodes = detect_particle_episodes(frame)
    assert len(episodes) == 2
    assert episodes.iloc[0]["duration_hours"] == 3
    assert episodes.iloc[0]["peak_pm10"] == 120
    assert episodes.iloc[1]["duration_hours"] == 3
    assert episodes.iloc[1]["start"] == index[8]


def test_monthly_summary_uses_observed_hours_only():
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"pm10": [50, 150, np.nan, 100], "dust": [5, 15, 20, 10]},
                         index=index)
    summary = monthly_summary(frame)
    january = summary.iloc[0]
    assert january["observed_hours"] == 3
    assert january["high_hours"] == 2
    assert january["high_percent"] == pytest.approx(200 / 3)
    assert january["pm10_mean"] == 100


def test_split_observed_forecast_does_not_show_future_as_measurement():
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    frame = pd.DataFrame({"pm10": range(5)}, index=index)
    observed, forecast = split_observed_forecast(frame, index[2] + pd.Timedelta(minutes=30))
    assert observed.index.tolist() == index[:3].tolist()
    assert forecast.index.tolist() == index[3:].tolist()


def test_best_outdoor_window_ranks_contiguous_forecast_periods():
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    frame = pd.DataFrame({
        "pm10": [80, 70, 30, 20, 40, 100],
        "pm2_5": [30, 25, 10, 8, 15, 40],
    }, index=index)
    result = best_outdoor_window(frame, duration_hours=2)
    assert result is not None
    assert result["start"] == index[2]
    assert result["end"] == index[4]
    assert result["mean_pm10"] == 25
    assert result["mean_pm2_5"] == 9


def test_best_outdoor_window_skips_missing_pm10_and_time_gaps():
    index = pd.DatetimeIndex([
        "2024-01-01 00:00", "2024-01-01 01:00",
        "2024-01-01 03:00", "2024-01-01 04:00",
    ])
    frame = pd.DataFrame({"pm10": [10, np.nan, 1, np.nan]}, index=index)
    assert best_outdoor_window(frame, duration_hours=2) is None


def test_inference_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Modèle absent"):
        load_forecaster(tmp_path / "missing.joblib")
