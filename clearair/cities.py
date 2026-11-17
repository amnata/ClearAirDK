"""Villes suivies par ClearAirDK (points de référence, pas des stations de mesure)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    slug: str
    lat: float
    lon: float
    region: str


CITIES = (
    City("Dakar", "dakar", 14.6937, -17.4441, "Dakar"),
    City("Thiès", "thies", 14.7910, -16.9359, "Thiès"),
    City("Touba", "touba", 14.8500, -15.8833, "Diourbel"),
    City("Diourbel", "diourbel", 14.6500, -16.2333, "Diourbel"),
    City("Saint-Louis", "saint-louis", 16.0179, -16.4896, "Saint-Louis"),
    City("Louga", "louga", 15.6167, -16.2167, "Louga"),
    City("Kaolack", "kaolack", 14.1652, -16.0726, "Kaolack"),
    City("Matam", "matam", 15.6559, -13.2554, "Matam"),
    City("Tambacounda", "tambacounda", 13.7707, -13.6673, "Tambacounda"),
    City("Kolda", "kolda", 12.8833, -14.9500, "Kolda"),
    City("Ziguinchor", "ziguinchor", 12.5833, -16.2719, "Ziguinchor"),
    City("Kédougou", "kedougou", 12.5579, -12.1743, "Kédougou"),
)

BY_NAME = {city.name: city for city in CITIES}
DEFAULT_CITY = CITIES[0]


def get(name: str) -> City:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Ville inconnue : {name}") from exc
