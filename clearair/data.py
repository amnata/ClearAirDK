"""Chargement, normalisation et collecte des observations."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from clearair import config as C


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise un tableau horaire en index datetime et colonnes internes."""
    frame = data.copy()
    if "time" in frame.columns:
        frame = frame.set_index("time")
    elif "datetime" in frame.columns:
        frame = frame.set_index("datetime")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(C.TIMEZONE).tz_localize(None)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame.index.name = "time"
    frame = frame[~frame.index.duplicated(keep="last")]

    for column in [*C.POLLUTANTS, "aqi", *C.WEATHER]:
        if column not in frame.columns:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] < 0, column] = float("nan")
    return frame[[*C.POLLUTANTS, "aqi", *C.WEATHER]]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Charge puis normalise un CSV horaire."""
    return clean(pd.read_csv(path))


def _hourly_frame(payload: dict[str, Any], mapping: dict[str, str]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Réponse Open-Meteo invalide : bloc hourly/time absent.")
    result = pd.DataFrame({"time": hourly["time"]})
    for api_name, column in mapping.items():
        if api_name in hourly:
            values = hourly[api_name]
            if len(values) != len(result):
                raise ValueError(f"Réponse Open-Meteo incohérente pour {api_name}.")
            result[column] = values
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    return result.set_index("time")


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=C.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Échec de la requête Open-Meteo : {exc}") from exc
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Réponse Open-Meteo invalide.")
    return payload


def fetch_history(
    start: date, end: date, lat: float = C.LATITUDE, lon: float = C.LONGITUDE
) -> pd.DataFrame:
    """Récupère qualité de l'air et météo historiques sur la période demandée."""
    if start > end:
        raise ValueError("La date de début doit précéder la date de fin.")
    base = {"latitude": lat, "longitude": lon, "timezone": C.TIMEZONE,
            "start_date": start.isoformat(), "end_date": end.isoformat()}
    air_payload = _get_json(
        C.AIR_QUALITY_URL,
        {**base, "hourly": ",".join(C.AIR_VARIABLES), "domains": "cams_global"},
    )
    weather_payload = _get_json(
        C.WEATHER_ARCHIVE_URL,
        {**base, "hourly": ",".join(C.WEATHER_VARIABLES)},
    )
    air = _hourly_frame(air_payload, C.AIR_VARIABLES)
    weather = _hourly_frame(weather_payload, C.WEATHER_VARIABLES)
    return clean(air.join(weather, how="outer").reset_index())


def fetch_recent(lat: float = C.LATITUDE, lon: float = C.LONGITUDE) -> pd.DataFrame:
    """Récupère observations récentes et prévisions horaires pour le tableau de bord."""
    base = {"latitude": lat, "longitude": lon, "timezone": C.TIMEZONE}
    air_payload = _get_json(
        C.AIR_QUALITY_URL,
        {**base, "hourly": ",".join(C.AIR_VARIABLES), "past_days": 7, "forecast_days": 3},
    )
    weather_payload = _get_json(
        C.WEATHER_FORECAST_URL,
        {**base, "hourly": ",".join(C.WEATHER_VARIABLES), "past_days": 7,
         "forecast_days": 3},
    )
    air = _hourly_frame(air_payload, C.AIR_VARIABLES)
    weather = _hourly_frame(weather_payload, C.WEATHER_VARIABLES)
    return clean(air.join(weather, how="outer").reset_index())


def fetch_all_cities(cities, workers: int = 6) -> dict[str, pd.DataFrame | None]:
    """Récupère en parallèle les données récentes de plusieurs villes.

    Une ville en échec vaut ``None`` : les autres restent utilisables.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(city):
        try:
            return city.name, fetch_recent(city.lat, city.lon)
        except (RuntimeError, ValueError):
            return city.name, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, cities))
