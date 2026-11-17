"""Configuration commune de ClearAirDK."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CITY = "Dakar"
LATITUDE = 14.6937
LONGITUDE = -17.4441
TIMEZONE = "Africa/Dakar"

CSV_PATH = DATA_DIR / "dakar_hourly.csv"
FORECAST_MODEL_PATH = MODEL_DIR / "forecast.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_VARIABLES = {
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "dust": "dust",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "us_aqi": "aqi",
}
WEATHER_VARIABLES = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "pressure_msl": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
}

POLLUTANTS = ["pm10", "pm2_5", "dust", "no2", "o3"]
WEATHER = ["temp", "rh", "pressure", "wind_speed", "wind_dir"]
HORIZONS = [6, 12, 24]
QUANTILES = (0.1, 0.5, 0.9)
TEST_FRACTION = 0.2
CALIB_FRACTION = 0.25  # part du bloc d'apprentissage réservée à la calibration
REQUEST_TIMEOUT_SECONDS = 20


# Seuils de référence OMS 2021 (moyennes sur 24 h), en µg/m³.
WHO_PM10_24H = 45.0
WHO_PM25_24H = 15.0
# Repère exploratoire ClearAirDK (voir README), distinct d'un seuil réglementaire.
EXPLORATORY_PM10 = 100.0


def paths_for(slug: str = "dakar") -> dict[str, Path]:
    """Chemins de données, modèles et rapports pour une ville (Dakar = noms historiques)."""
    if slug == "dakar":
        return {"csv": CSV_PATH, "model": FORECAST_MODEL_PATH,
                "anomaly": ANOMALY_MODEL_PATH, "metrics": METRICS_PATH}
    return {
        "csv": DATA_DIR / f"{slug}_hourly.csv",
        "model": MODEL_DIR / f"forecast_{slug}.joblib",
        "anomaly": MODEL_DIR / f"anomaly_{slug}.joblib",
        "metrics": REPORT_DIR / f"metrics_{slug}.json",
    }
