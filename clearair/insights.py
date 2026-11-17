"""Statistiques descriptives des épisodes et tendances de particules."""

from __future__ import annotations

import pandas as pd

from clearair import config as C


def split_observed_forecast(
    data: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les lignes passées des prévisions au moment donné."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Les données doivent avoir un index datetime.")
    cutoff = pd.Timestamp(as_of)
    if data.index.tz is not None:
        cutoff = cutoff.tz_localize(data.index.tz) if cutoff.tzinfo is None else cutoff.tz_convert(data.index.tz)
    elif cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(C.TIMEZONE).tz_localize(None)
    ordered = data.sort_index()
    return ordered.loc[ordered.index <= cutoff], ordered.loc[ordered.index > cutoff]


def best_outdoor_window(
    forecast: pd.DataFrame,
    duration_hours: int,
) -> dict[str, object] | None:
    """Finds the lowest mean-PM10 contiguous forecast interval of requested length."""
    if not isinstance(forecast.index, pd.DatetimeIndex):
        raise TypeError("Les prévisions doivent avoir un index datetime.")
    if duration_hours < 1:
        raise ValueError("La durée doit être d'au moins une heure.")
    if "pm10" not in forecast:
        raise ValueError("La colonne pm10 est requise.")

    frame = forecast.sort_index()
    pm10 = pd.to_numeric(frame["pm10"], errors="coerce")
    pm25 = (
        pd.to_numeric(frame["pm2_5"], errors="coerce")
        if "pm2_5" in frame else pd.Series(float("nan"), index=frame.index)
    )
    best: dict[str, object] | None = None
    best_key: tuple[float, float] | None = None
    for start in range(len(frame) - duration_hours + 1):
        end = start + duration_hours
        interval = frame.index[start:end]
        if len(interval) != duration_hours:
            continue
        if duration_hours > 1 and not (interval[1:] - interval[:-1] == pd.Timedelta(hours=1)).all():
            continue
        pm10_values = pm10.iloc[start:end]
        if pm10_values.isna().any():
            continue
        candidate = {
            "start": interval[0],
            "end": interval[-1] + pd.Timedelta(hours=1),
            "duration_hours": duration_hours,
            "mean_pm10": float(pm10_values.mean()),
            "peak_pm10": float(pm10_values.max()),
            "mean_pm2_5": float(pm25.iloc[start:end].mean()),
        }
        mean_pm25 = candidate["mean_pm2_5"]
        key = (
            float(candidate["mean_pm10"]),
            float(mean_pm25) if pd.notna(mean_pm25) else float("inf"),
        )
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
    return best


def monthly_summary(data: pd.DataFrame, threshold: float = 100.0) -> pd.DataFrame:
    """Résume PM10 et poussière par mois, sans imputer les heures manquantes."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Les données doivent avoir un index datetime.")
    if "pm10" not in data:
        raise ValueError("La colonne pm10 est requise.")

    pm10 = pd.to_numeric(data["pm10"], errors="coerce").dropna().sort_index()
    if pm10.empty:
        return pd.DataFrame(
            columns=["pm10_mean", "pm10_median", "observed_hours",
                     "high_hours", "high_percent", "dust_mean"]
        )

    monthly = pm10.resample("MS").agg(
        pm10_mean="mean",
        pm10_median="median",
        observed_hours="count",
    )
    monthly["high_hours"] = (
        pm10.ge(threshold).astype(int).resample("MS").sum().reindex(monthly.index, fill_value=0)
    )
    monthly["high_percent"] = 100 * monthly["high_hours"] / monthly["observed_hours"]
    if "dust" in data:
        dust = pd.to_numeric(data["dust"], errors="coerce")
        monthly["dust_mean"] = dust.resample("MS").mean().reindex(monthly.index)
    else:
        monthly["dust_mean"] = float("nan")
    return monthly


def detect_particle_episodes(
    data: pd.DataFrame,
    threshold: float = 100.0,
    min_hours: int = 3,
) -> pd.DataFrame:
    """Trouve des séries d'heures PM10 élevées, séparées par une heure non élevée."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Les données doivent avoir un index datetime.")
    if "pm10" not in data:
        raise ValueError("La colonne pm10 est requise.")
    if min_hours < 1:
        raise ValueError("min_hours doit être supérieur ou égal à 1.")

    pm10 = pd.to_numeric(data["pm10"], errors="coerce").dropna().sort_index()
    high = pm10.loc[pm10 >= threshold]
    columns = ["start", "end", "duration_hours", "peak_pm10", "mean_pm10", "mean_dust"]
    if high.empty:
        return pd.DataFrame(columns=columns)

    groups = high.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    episodes = []
    dust = (
        pd.to_numeric(data["dust"], errors="coerce")
        if "dust" in data else pd.Series(dtype=float, index=data.index)
    )
    for _, values in high.groupby(groups.to_numpy()):
        if len(values) < min_hours:
            continue
        interval = dust.reindex(values.index)
        episodes.append({
            "start": values.index[0],
            "end": values.index[-1],
            "duration_hours": int(len(values)),
            "peak_pm10": float(values.max()),
            "mean_pm10": float(values.mean()),
            "mean_dust": float(interval.mean()) if interval.notna().any() else float("nan"),
        })
    return pd.DataFrame(episodes, columns=columns)


def city_snapshot(data: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, object] | None:
    """Résume une ville : dernier niveau observé et prévisions des prochaines 24 h."""
    from clearair.advice import level_from

    observed, future = split_observed_forecast(data, as_of)
    obs = observed.dropna(subset=["pm10", "pm2_5"], how="all")
    if obs.empty:
        return None
    last = obs.iloc[-1]
    pm10 = float(last["pm10"]) if pd.notna(last["pm10"]) else None
    pm25 = float(last["pm2_5"]) if pd.notna(last["pm2_5"]) else None
    nxt = future.head(24)["pm10"].dropna() if not future.empty else pd.Series(dtype=float)
    return {
        "pm10": pm10, "pm2_5": pm25, "level": level_from(pm10, pm25),
        "observed_at": obs.index[-1],
        "next24_mean": float(nxt.mean()) if not nxt.empty else None,
        "next24_max": float(nxt.max()) if not nxt.empty else None,
        "next24_peak_time": nxt.idxmax() if not nxt.empty else None,
    }
