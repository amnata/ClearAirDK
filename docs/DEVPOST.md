# Devpost : texte à coller (English)

**Project name:** ClearAirDK, dust you can act on
**Track:** Sustainability
**Tagline:** From raw dust numbers to decisions: when to go out, work and play, across 12 Senegalese cities.

## Inspiration
Every dry season, harmattan dust degrades air across Senegal. Children, people with asthma, older adults and outdoor workers have no simple tool that says *what to do and when*. Micrograms per cubic metre do not help a parent deciding about school recess.

## Why Sustainability
Dust and air quality are an environmental challenge made worse by drought and a changing climate. ClearAirDK helps communities *live well* with it: understand the environment, anticipate it, and reduce exposure. Health is the use case, the environment is the problem.

## What it does
- Live PM10/PM2.5 and profile-based advice for 12 cities, with a map of the country.
- A planner for three real decisions: an outing, an outdoor work shift (exposure avoided versus the usual schedule), and a school day (hour-by-hour traffic light for recess and sport).
- A forecast (+6/12/24 h) with calibrated uncertainty and a probability of exceeding 100 µg/m³.
- A Groq-powered assistant (optional; free-tier API, subject to quotas) with a safe rule-based fallback; emergencies never go through the LLM.
- WhatsApp-ready alerts in French and English, because most people will never open a dashboard.

## How we built it
Python, Streamlit, Plotly, scikit-learn (gradient boosting quantile models, conformal calibration, classifier, Isolation Forest), Open-Meteo/CAMS data, Groq API (optional). 28 automated tests.

## How we used AI
In the product: locally trained scikit-learn models (forecast, exceedance probability, anomaly detection) and an optional assistant powered by the Groq API (openai/gpt-oss-120b) that only receives air-quality context and a chosen profile. Development: AI coding assistants (Claude and GitHub Copilot) helped write code, tests and docs; everything was reviewed and run by us.

## Challenges
Our first model barely beat the persistence baseline and its 80% interval covered only about 65% of cases. We added chronological calibration (conformal), reported the modest gain honestly, and found the strongest value in exceedance probabilities (Brier 0.11 vs 0.25 for climatology at +6 h).

## What we learned
Evaluation design matters more than model complexity: time-based splits, baselines, calibration, and saying what the model cannot do.

## What's next
Wolof and other national languages validated with speakers, local low-cost sensors, SMS alerts, and a pilot with a school and a health centre in Thiès.

## Honest limitations
Modelled data (no local sensor), one reference point per city, indicative advice, not a medical device.
