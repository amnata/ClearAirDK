"""CLI pour collecter les données et entraîner les modèles ClearAirDK."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from clearair import cities as CITIES
from clearair import config as C
from clearair.data import fetch_history, load_csv
from clearair.models import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="Dakar",
                        help="Ville suivie (voir clearair/cities.py), Dakar par défaut.")
    parser.add_argument("--csv", help="CSV horaire à utiliser (time, pm10, ...).")
    parser.add_argument("--download", action="store_true", help="Télécharger l'historique Open-Meteo.")
    parser.add_argument("--start", default="2022-08-01", help="Début inclus, format AAAA-MM-JJ.")
    parser.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(),
                        help="Fin incluse, format AAAA-MM-JJ.")
    args = parser.parse_args()
    city = CITIES.get(args.city)
    paths = C.paths_for(city.slug)

    if args.csv and args.download:
        parser.error("Choisissez --csv ou --download, pas les deux.")
    if args.download:
        data = fetch_history(date.fromisoformat(args.start), date.fromisoformat(args.end),
                             city.lat, city.lon)
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(paths["csv"], index_label="time")
        print(f"Données sauvegardées dans {paths['csv']} ({len(data)} lignes).")
    elif args.csv:
        data = load_csv(args.csv)
    elif paths["csv"].is_file():
        data = load_csv(paths["csv"])
    else:
        parser.error("Indiquez --csv ou --download, ou placez data/dakar_hourly.csv.")
    print(
        "Entraînement des prévisions (+6 h, +12 h, +24 h) et du détecteur "
        "d'anomalies…",
        flush=True,
    )
    report = train_all(data, model_path=paths["model"], metrics_path=paths["metrics"],
                       anomaly_path=paths["anomaly"], city=city.name)
    for horizon, metrics in report["horizons"].items():
        print(
            f"+{horizon} h | modèle MAE={metrics['model']['mae']:.1f} "
            f"| persistance MAE={metrics['baseline']['mae']:.1f} "
            f"| couverture 80%={metrics['coverage_80']:.1%}"
            f"| gain vs persistance={metrics['skill_vs_persistence']:+.1%}"
        )
    print(f"Rapport : {paths['metrics']}")


if __name__ == "__main__":
    main()
