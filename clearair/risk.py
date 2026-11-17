"""Analyses d'exposition : repères OMS, rythme journalier, horaires de travail, école."""

from __future__ import annotations

import pandas as pd

from clearair import config as C
from clearair.advice import LEVELS, level_from
from clearair.insights import best_outdoor_window

# Facteurs relatifs de ventilation (repos = 1). Ordres de grandeur, pas une dose clinique.
ACTIVITY_FACTORS = {
    "Repos / travail assis": 1.0,
    "Marche, travail léger": 2.0,
    "Travail physique, sport": 4.0,
}


def daily_means(data: pd.DataFrame, column: str = "pm10", min_hours: int = 18) -> pd.Series:
    """Moyennes journalières, uniquement pour les jours assez complets."""
    series = pd.to_numeric(data[column], errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype=float)
    grouped = series.resample("D")
    means, counts = grouped.mean(), grouped.count()
    return means[counts >= min_hours].dropna()


def who_exceedance(data: pd.DataFrame) -> pd.DataFrame:
    """Par année : jours valides et part des jours au-dessus du repère OMS 2021 (24 h)."""
    rows = []
    for column, limit, key in (("pm10", C.WHO_PM10_24H, "pm10"), ("pm2_5", C.WHO_PM25_24H, "pm25")):
        if column not in data:
            continue
        daily = daily_means(data, column)
        if daily.empty:
            continue
        by_year = daily.groupby(daily.index.year)
        for year, values in by_year:
            rows.append({
                "year": int(year), "pollutant": key, "valid_days": int(len(values)),
                "days_over": int((values > limit).sum()),
                "share_over": float((values > limit).mean()),
                "annual_mean": float(values.mean()), "limit_24h": limit,
            })
    return pd.DataFrame(rows, columns=["year", "pollutant", "valid_days", "days_over",
                                       "share_over", "annual_mean", "limit_24h"])


def hourly_profile(data: pd.DataFrame, months: list[int] | None = None,
                   column: str = "pm10") -> pd.Series:
    """Moyenne par heure du jour (0-23), éventuellement limitée à certains mois."""
    series = pd.to_numeric(data[column], errors="coerce").dropna()
    if months:
        series = series[series.index.month.isin(months)]
    if series.empty:
        return pd.Series(dtype=float)
    return series.groupby(series.index.hour).mean().reindex(range(24))


def month_hour_matrix(data: pd.DataFrame, column: str = "pm10") -> pd.DataFrame:
    """Matrice mois × heure des moyennes (pour une carte de chaleur)."""
    series = pd.to_numeric(data[column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame()
    matrix = series.groupby([series.index.month, series.index.hour]).mean().unstack()
    return matrix.reindex(index=range(1, 13), columns=range(24))


def seasonal_contrast(data: pd.DataFrame) -> dict[str, float]:
    """Moyenne PM10 en saison harmattan (nov.-mars) et hors saison."""
    series = pd.to_numeric(data["pm10"], errors="coerce").dropna()
    dry = series[series.index.month.isin([11, 12, 1, 2, 3])]
    other = series[~series.index.month.isin([11, 12, 1, 2, 3])]
    if dry.empty or other.empty:
        return {}
    return {"harmattan_mean": float(dry.mean()), "other_mean": float(other.mean()),
            "ratio": float(dry.mean() / other.mean())}


def exposure_index(window: pd.DataFrame, activity: str) -> float:
    """Indice relatif d'exposition : somme PM10 × facteur d'activité sur les heures."""
    if activity not in ACTIVITY_FACTORS:
        raise ValueError(f"Activité inconnue : {activity}")
    return float(pd.to_numeric(window["pm10"], errors="coerce").sum() * ACTIVITY_FACTORS[activity])


def best_window_within(forecast: pd.DataFrame, duration: int,
                       earliest: int = 6, latest: int = 19):
    """Meilleur créneau continu dont toutes les heures sont dans [earliest, latest)."""
    if earliest >= latest:
        raise ValueError("L'heure de début doit précéder l'heure de fin.")
    mask = (forecast.index.hour >= earliest) & (forecast.index.hour < latest)
    return best_outdoor_window(forecast.loc[mask], duration)


def compare_schedule(day_forecast: pd.DataFrame, duration: int, usual_start: int,
                     activity: str, earliest: int = 6, latest: int = 19) -> dict | None:
    """Compare l'horaire habituel au meilleur horaire : gain d'exposition relatif."""
    best = best_window_within(day_forecast, duration, earliest, latest)
    usual_rows = day_forecast.loc[(day_forecast.index.hour >= usual_start)
                                  & (day_forecast.index.hour < usual_start + duration)]
    if best is None or len(usual_rows) != duration or usual_rows["pm10"].isna().any():
        return None
    best_rows = day_forecast.loc[best["start"]: best["end"] - pd.Timedelta(hours=1)]
    usual_idx = exposure_index(usual_rows, activity)
    best_idx = exposure_index(best_rows, activity)
    gain = 0.0 if usual_idx <= 0 else max(0.0, 100 * (usual_idx - best_idx) / usual_idx)
    return {"best": best, "usual_mean_pm10": float(usual_rows["pm10"].mean()),
            "usual_index": usual_idx, "best_index": best_idx, "reduction_percent": gain}


def school_plan(day_forecast: pd.DataFrame, start_hour: int = 8, end_hour: int = 17) -> pd.DataFrame:
    """Feu tricolore horaire pour récréation et sport à l'école."""
    rows = []
    sub = day_forecast.loc[(day_forecast.index.hour >= start_hour)
                           & (day_forecast.index.hour < end_hour)]
    for stamp, row in sub.iterrows():
        pm10 = row["pm10"] if pd.notna(row["pm10"]) else None
        pm25 = row.get("pm2_5") if pd.notna(row.get("pm2_5")) else None
        level = level_from(pm10, pm25)
        if level is None:
            plan, light = "Données manquantes", "gris"
        elif level <= 1:
            plan, light = "Récréation et sport normaux", "vert"
        elif level == 2:
            plan, light = "Jeux calmes, pas de sport intense", "orange"
        else:
            plan, light = "Activités en intérieur", "rouge"
        rows.append({"Heure": f"{stamp:%H:%M}", "PM10": pm10, "Niveau": None if level is None
                     else LEVELS[level].label, "Conseil école": plan, "Feu": light})
    return pd.DataFrame(rows, columns=["Heure", "PM10", "Niveau", "Conseil école", "Feu"])
