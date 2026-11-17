# ClearAirDK 🌬️

**EN summary.** ClearAirDK turns dust and air-quality data for 12 Senegalese cities into
*decisions*: when to go out, when to schedule outdoor work, when school children can play
outside. It forecasts PM10 (+6/12/24 h) with calibrated uncertainty and a probability of
exceeding 100 µg/m³, answers questions through an optional Groq assistant (with a safe
offline fallback), generates WhatsApp-ready alerts in French/English, **Track: Sustainability** (living well in a changing environment; health impact is the use case).

> **Prototype, pas dispositif médical.** Les données Open-Meteo (modèle CAMS) sont modélisées,
> pas mesurées par un capteur local. Les conseils sont indicatifs.

## Le problème

Chaque saison sèche (harmattan, environ novembre à mars), la poussière dégrade l'air au Sénégal.
Les plus exposés (enfants, personnes asthmatiques, personnes âgées, travailleurs en extérieur)
ont peu d'outils simples, locaux et en français pour savoir **quand** agir. Les chiffres bruts
(µg/m³) ne disent pas quoi faire.

## Ce que fait ClearAirDK

| Onglet | Utilité concrète |
|---|---|
| **Maintenant** | PM10/PM2.5, niveau indicatif, conseil par profil, gestes de protection (dont le bon type de masque), prévision ML avec **risque de dépasser 100 µg/m³** |
| **Planifier** | 3 modes : *sortie* (meilleur créneau), *travail en extérieur* (gain d'exposition vs horaire habituel), *école* (feu tricolore horaire pour récréation et sport) |
| **Sénégal** | Carte et classement de 12 villes en direct, avec pic prévu à 24 h |
| **Assistant IA** | Questions en langage naturel (Groq si clé API, sinon règles prudentes). Les urgences ne passent jamais par le LLM |
| **Partager** | Bulletin court FR/EN prêt pour WhatsApp : l'information atteint ceux qui n'ouvriront pas un tableau de bord |
| **Historique** | Jours au-dessus du repère OMS 2021, heures les plus chargées, carte de chaleur mois × heure, épisodes |
| **À propos** | Évaluation chiffrée et honnête du modèle, limites |

## Rigueur scientifique

- **Split chronologique** apprentissage / calibration / test (jamais de mélange aléatoire).
- **Baseline de persistance** : le modèle est comparé à « PM10 reste ce qu'il est ».
- **Intervalles recalibrés (CQR, conformal)** : couverture de l'intervalle à 80 % passée d'environ
  62-69 % (brut) à environ 77-80 % sur le bloc de test.
- **Probabilité de dépassement** (PM10 ≥ 100 µg/m³) : le résultat le plus utile. Sur le test,
  Brier 0,11 / 0,16 / 0,18 à +6/12/24 h contre 0,25 pour la climatologie ; ROC AUC 0,92 / 0,85 / 0,79.
- **Gain modeste sur la valeur exacte** : 5 à 11 % de MAE en moins que la persistance. Nous le
  disons plutôt que de le cacher ; l'évaluation par saison est dans `reports/metrics.json`.
- **Explicabilité** : importance par permutation (variables qui pèsent à +24 h).

(Chiffres obtenus avec `python train.py --csv data/dakar_hourly.csv` ; relancez-le pour
reproduire vos propres valeurs.)

## Utilisation de l'IA (divulgation)

- **IA dans le produit** : modèles scikit-learn entraînés localement (prévision, classification,
  détection d'anomalies) et assistant conversationnel via l'API **Groq** (facultatif, offre gratuite
  soumise aux quotas ; modèle `openai/gpt-oss-120b`). Sans clé, des règles locales prudentes répondent.
- **Assistants de codage** : le code, les tests et la documentation ont été écrits avec l'aide
  d'assistants IA (Claude d'Anthropic ; GitHub Copilot dans VS Code pour la version initiale).
  Tout le code a été relu, exécuté et testé par l'équipe.
- **Garde-fous** : l'assistant ne reçoit que le contexte de qualité de l'air et le profil choisi,
  ne diagnostique pas, et renvoie vers les urgences en cas de symptômes graves.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Démarrage rapide

```bash
python train.py --csv data/dakar_hourly.csv           # Dakar, avec CSV fourni
python train.py --city Thiès --download --start 2022-08-01   # autre ville (Internet requis)
streamlit run app.py
```

Assistant IA (facultatif) : créez une clé sur GroqCloud puis définissez
`export GROQ_API_KEY=...` ou saisissez-la dans la barre latérale. Modèle changeable via
`CLEARAIR_LLM_MODEL` (par défaut `openai/gpt-oss-120b`).

## Architecture

```text
app.py                  Tableau de bord Streamlit (7 onglets)
train.py                Collecte Open-Meteo + entraînement (option --city)
clearair/
  cities.py             12 villes suivies
  data.py               Lecture, nettoyage, API Open-Meteo (parallèle pour les villes)
  features.py           Retards, moyennes passées, saison harmattan
  models.py             Quantiles + conformal, classifieur de dépassement, évaluation, importance
  insights.py           Créneaux, épisodes, tendances, résumé par ville
  risk.py               Repères OMS, rythme horaire, horaires de travail, plan école
  advice.py             Niveaux, conseils FR/EN, gestes pratiques
  alerts.py             Messages WhatsApp/SMS
  assistant.py          Groq + repli hors-ligne + garde-fous
  anomaly.py            Valeurs atypiques (Isolation Forest)
tests/                  tests (python -m pytest)
docs/                   Texte Devpost et script de démo
```

## Limites

- Un point de référence par ville, pas de capteur local : à valider avec des mesures de terrain.
- Une valeur élevée n'identifie pas à elle seule l'origine saharienne.
- L'air intérieur et le trajet ne sont pas pris en compte.
- Les modèles des villes autres que Dakar demandent un téléchargement et un entraînement préalables.

## Prochaines étapes

Wolof et autres langues nationales validées avec des locuteurs, intégration de capteurs
low-cost locaux, alertes SMS pour les zones à faible connectivité, pilote avec une école et un
centre de santé de Thiès.
