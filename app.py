"""ClearAirDK : comprendre la poussière, protéger les plus fragiles, agir au Sénégal."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from clearair import alerts, assistant, cities as CITY, risk
from clearair import config as C
from clearair.advice import LEVELS, PROFILES, advice_for, level_from, practical_tips
from clearair.data import fetch_all_cities, fetch_recent, load_csv
from clearair.features import build_features
from clearair.insights import (
    best_outdoor_window, city_snapshot, detect_particle_episodes,
    monthly_summary, split_observed_forecast,
)
from clearair.models import load_forecaster, predict

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre")
LIGHTS = {"vert": "🟢", "orange": "🟠", "rouge": "🔴", "gris": "⚪"}


def fmt_day(day) -> str:
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def badge(level: int | None) -> str:
    if level is None:
        return "⚪ Inconnu"
    return f"{alerts.EMOJI[level]} {LEVELS[level].label}"


st.set_page_config(page_title="ClearAirDK · Poussière et santé au Sénégal",
                   page_icon="🌬️", layout="wide")
st.title("🌬️ ClearAirDK")
st.subheader("Voir la poussière venir. Protéger les plus fragiles. Agir au bon moment.")
st.warning("Prototype de sensibilisation, pas un service d'alerte ni un dispositif médical. "
           "Les données (Open-Meteo / CAMS) sont modélisées, pas mesurées par un capteur local.")

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Votre contexte")
    city = CITY.get(st.selectbox("Ville", [c.name for c in CITY.CITIES]))
    profile_key = st.selectbox("Profil", list(PROFILES), format_func=PROFILES.get)
    st.caption("Le profil adapte les conseils, jamais les mesures.")
    refresh = st.button("Actualiser les données")
    st.divider()

paths = C.paths_for(city.slug)


@st.cache_data(ttl=900, show_spinner=False)
def get_recent(name: str) -> pd.DataFrame:
    c = CITY.get(name)
    return fetch_recent(c.lat, c.lon)


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(path: str) -> pd.DataFrame:
    return load_csv(path)


@st.cache_data(ttl=900, show_spinner=False)
def get_all_cities() -> dict:
    return fetch_all_cities(CITY.CITIES)


if refresh:
    get_recent.clear()
    get_all_cities.clear()

frame, source = None, "Open-Meteo (observations modélisées CAMS et prévisions)"
try:
    with st.spinner("Chargement des données…"):
        frame = get_recent(city.name)
except (RuntimeError, ValueError) as exc:
    st.error(f"Données en ligne indisponibles : {exc}")
    if paths["csv"].is_file():
        st.info("Utilisation de l'historique local, sans prévisions en direct.")
        frame, source = get_history(str(paths["csv"])), "Historique local"
if frame is None or frame.empty:
    st.stop()

history = None
if paths["csv"].is_file():
    try:
        history = get_history(str(paths["csv"])).dropna(subset=["pm10"])
    except (OSError, ValueError, pd.errors.ParserError):
        history = None

now = pd.Timestamp.now(tz=C.TIMEZONE).tz_localize(None)
observed, future = split_observed_forecast(frame, now)
obs_p = observed.dropna(subset=["pm10", "pm2_5"], how="all")
future_p = future.dropna(subset=["pm10", "pm2_5"], how="all")
latest = obs_p.iloc[-1] if not obs_p.empty else None
pm10 = float(latest["pm10"]) if latest is not None and pd.notna(latest["pm10"]) else None
pm25 = float(latest["pm2_5"]) if latest is not None and pd.notna(latest["pm2_5"]) else None
level = level_from(pm10, pm25)
age_h = max(0.0, (now - latest.name).total_seconds() / 3600) if latest is not None else None
fresh = age_h is not None and age_h <= 3

# Prévision ML de la ville (si le modèle existe)
ml, ml_msg = None, None
if fresh:
    try:
        ml = predict(load_forecaster(paths["model"]), build_features(observed.loc[:latest.name]))
    except FileNotFoundError:
        ml_msg = "no_model"
    except (ValueError, KeyError) as exc:
        ml_msg = f"Prévision ClearAirDK indisponible : {exc}"

next24 = future_p.head(24)
peak_time = next24["pm10"].idxmax() if not next24["pm10"].dropna().empty else None
best_today = risk.best_window_within(future_p.head(48), 2, 6, 19) if not future_p.empty else None
window_text = (f"Créneau prévu le moins chargé : {best_today['start']:%d/%m %H:%M}–"
               f"{best_today['end']:%H:%M} (PM10 ≈ {best_today['mean_pm10']:.0f} µg/m³)."
               if best_today else "")
peak_text = f"Pic de PM10 prévu vers {peak_time:%d/%m %H:%M}." if peak_time is not None else ""

tabs = st.tabs(["Maintenant", "Planifier", "Sénégal", "Assistant IA", "Partager",
                "Historique", "À propos"])
t_now, t_plan, t_sn, t_ai, t_share, t_hist, t_about = tabs

# ---------------------------------------------------------------- Maintenant
with t_now:
    st.header(f"{city.name} : la situation")
    st.caption(f"Source : {source}. Heures de Dakar (GMT).")
    if latest is None:
        st.warning("Aucune mesure passée disponible.")
    else:
        st.caption(f"Dernière donnée : **{latest.name:%d/%m/%Y %H:%M}** (il y a ≈ {age_h:.0f} h).")
        if not fresh:
            st.warning("Donnée ancienne : ne la considérez pas comme la situation actuelle.")
        a, b, c = st.columns(3)
        a.metric("PM10", f"{pm10:.0f} µg/m³" if pm10 is not None else "n/d")
        b.metric("PM2.5", f"{pm25:.0f} µg/m³" if pm25 is not None else "n/d")
        c.metric("Niveau indicatif", badge(level))
        st.markdown(f"**Conseil · {PROFILES[profile_key].lower()}**")
        st.info(advice_for(profile_key, level))
        tips = practical_tips(level)
        if tips:
            with st.expander("Gestes simples pour réduire l'exposition"):
                for tip in tips:
                    st.markdown(f"- {tip}")
        if pm10 is not None and pm10 > C.WHO_PM10_24H:
            st.caption(f"Repère OMS 2021 (moyenne sur 24 h) : PM10 ≤ {C.WHO_PM10_24H:.0f} µg/m³. "
                       "La valeur ci-dessus est horaire, la comparaison est donc indicative.")

    parts = []
    hist48 = observed.tail(48).dropna(subset=["pm10"])
    if not hist48.empty:
        parts.append(pd.DataFrame({"Heure": hist48.index, "PM10": hist48["pm10"].to_numpy(),
                                   "Période": "Observé"}))
    fc = future_p.head(36).dropna(subset=["pm10"])
    if not fc.empty:
        parts.append(pd.DataFrame({"Heure": fc.index, "PM10": fc["pm10"].to_numpy(),
                                   "Période": "Prévision Open-Meteo"}))
    if parts:
        fig = px.line(pd.concat(parts), x="Heure", y="PM10", color="Période", markers=True,
                      color_discrete_map={"Observé": "#26734d", "Prévision Open-Meteo": "#3274a1"},
                      labels={"PM10": "PM10 (µg/m³)"})
        fig.add_hline(y=C.EXPLORATORY_PM10, line_dash="dot", annotation_text="repère 100 µg/m³")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prévision expérimentale ClearAirDK (modèle entraîné localement)")
    if ml:
        rows = []
        for h, v in sorted(ml.items()):
            p = v.get("p_exceed")
            rows.append({"Dans": f"{h} h", "PM10 médian": f"{v['median']:.0f} µg/m³",
                         "Intervalle ≈80 %": f"{v['lower']:.0f} à {v['upper']:.0f}",
                         "Risque de dépasser 100 µg/m³": f"{100 * p:.0f} %" if p is not None else "n/d"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Le « risque » est une probabilité estimée par le modèle pour cette heure précise ; "
                   "elle est plus fiable que la valeur exacte de PM10 (voir l'onglet À propos).")
    elif ml_msg == "no_model":
        if paths["csv"].is_file():
            st.info(f"Aucun modèle entraîné pour {city.name}.")
            if st.button("Entraîner le modèle maintenant (≈ 1 min)"):
                from clearair.models import train_all
                with st.spinner("Entraînement en cours…"):
                    train_all(get_history(str(paths["csv"])), model_path=paths["model"],
                              metrics_path=paths["metrics"], anomaly_path=paths["anomaly"],
                              city=city.name)
                st.rerun()
        else:
            st.info(f"La prévision ClearAirDK par apprentissage automatique est activée pour "
                    f"Dakar, la ville pilote. Pour {city.name}, utilisez la prévision "
                    f"Open-Meteo du graphique ci-dessus. L'extension à d'autres villes se fait "
                    f"avec : `python train.py --city {city.name} --download`.")
    else:
        st.info(ml_msg or "Prévision masquée : la dernière donnée est trop ancienne.")

# ---------------------------------------------------------------- Planifier
with t_plan:
    st.header("Choisir le bon moment")
    if future_p.empty:
        st.info("Pas de prévision horaire : connectez-vous à Internet.")
    else:
        mode = st.radio("Je planifie…", ["Une sortie", "Mon travail en extérieur", "La journée d'école"],
                        horizontal=True)
        days = sorted(set(future_p.index.date))
        day = st.selectbox("Jour", days, format_func=fmt_day)
        day_fc = future_p.loc[[s.date() == day for s in future_p.index]]
        if mode == "Une sortie":
            dur = st.selectbox("Durée (h)", [1, 2, 3, 4], index=1)
            w = risk.best_window_within(day_fc, dur, 6, 19)
            if w is None:
                st.warning("Données insuffisantes pour ce jour.")
            else:
                st.success(f"Meilleur créneau : **{w['start']:%H:%M}–{w['end']:%H:%M}** "
                           f"(PM10 moyen prévu {w['mean_pm10']:.0f} µg/m³).")
                st.info(advice_for(profile_key, level_from(w["mean_pm10"], w["mean_pm2_5"])))
        elif mode == "Mon travail en extérieur":
            c1, c2, c3 = st.columns(3)
            dur = c1.slider("Heures de tâche exposée", 1, 8, 4)
            usual = c2.slider("Heure de début habituelle", 6, 16, 9)
            act = c3.selectbox("Effort", list(risk.ACTIVITY_FACTORS))
            cmp_ = risk.compare_schedule(day_fc, dur, usual, act)
            if cmp_ is None:
                st.warning("Données insuffisantes (ou plage hors prévisions) pour comparer.")
            else:
                best = cmp_["best"]
                st.success(f"Début conseillé : **{best['start']:%H:%M}** → {best['end']:%H:%M}.")
                m1, m2, m3 = st.columns(3)
                m1.metric("PM10 horaire habituel", f"{cmp_['usual_mean_pm10']:.0f} µg/m³")
                m2.metric("PM10 avec ce créneau", f"{best['mean_pm10']:.0f} µg/m³")
                m3.metric("Exposition relative évitée", f"{cmp_['reduction_percent']:.0f} %")
                st.caption("L'indice relatif multiplie PM10 par un facteur d'effort (repos = 1). "
                           "C'est un ordre de grandeur pour comparer deux horaires, pas une dose médicale.")
        else:
            plan = risk.school_plan(day_fc)
            if plan.empty:
                st.warning("Pas de prévision pour les heures de classe.")
            else:
                shown = plan.copy()
                shown["Feu"] = shown["Feu"].map(LIGHTS)
                st.dataframe(shown, hide_index=True, use_container_width=True)
                reds = int((plan["Feu"] == "rouge").sum())
                st.info("Aucune heure rouge : activités normales." if reds == 0 else
                        f"{reds} heure(s) rouge(s) : prévoyez des activités en intérieur sur ce créneau.")
        st.caption("Comparaison relative des prévisions disponibles. Ne tient pas compte du trajet ni de l'air intérieur.")

# ---------------------------------------------------------------- Sénégal
with t_sn:
    st.header("Le Sénégal d'un coup d'œil")
    st.write("Douze villes comparées en direct : la poussière ne frappe pas partout de la même façon.")
    with st.spinner("Chargement des villes…"):
        all_data = get_all_cities()
    rows = []
    for c in CITY.CITIES:
        snap = city_snapshot(all_data[c.name], now) if all_data.get(c.name) is not None else None
        if snap:
            rows.append({"Ville": c.name, "lat": c.lat, "lon": c.lon, "PM10": snap["pm10"],
                         "Niveau": badge(snap["level"]), "level_code": snap["level"],
                         "Pic prévu (24 h)": snap["next24_max"],
                         "Heure du pic": snap["next24_peak_time"]})
    if not rows:
        st.warning("Aucune ville disponible pour le moment.")
    else:
        sn = pd.DataFrame(rows)
        sn["Taille"] = sn["PM10"].fillna(10).clip(lower=10)
        map_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
        kw = {"map_style": "open-street-map"} if hasattr(px, "scatter_map") else {"mapbox_style": "open-street-map"}
        fig = map_fn(sn, lat="lat", lon="lon", size="Taille", color="PM10", hover_name="Ville",
                     hover_data={"lat": False, "lon": False, "Taille": False, "PM10": ":.0f"},
                     color_continuous_scale=["#2e8b57", "#d4a017", "#e87524", "#c83232", "#742b8a"],
                     range_color=(0, 250), zoom=5.2, height=480,
                     center={"lat": 14.3, "lon": -15.0}, **kw)
        st.plotly_chart(fig, use_container_width=True)
        table = sn.sort_values("PM10", ascending=False)[["Ville", "PM10", "Niveau", "Pic prévu (24 h)", "Heure du pic"]]
        table["Heure du pic"] = pd.to_datetime(table["Heure du pic"]).dt.strftime("%d/%m %H:%M")
        st.dataframe(table.round(0), hide_index=True, use_container_width=True)
        worst = sn.sort_values("PM10", ascending=False).iloc[0]
        st.caption(f"Ville la plus chargée actuellement : **{worst['Ville']}**. "
                   "Chaque ville est un point de référence de modèle, pas une station.")

# ---------------------------------------------------------------- Assistant IA
# CHANGÉ : key_input est maintenant lu depuis les secrets (le champ a été supprimé)
try:
    key_input = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    key_input = os.getenv("GROQ_API_KEY")

with t_ai:
    st.header("Posez votre question")
    use_llm = assistant.llm_available(key_input or None)
    # CHANGÉ : on ne parle plus de la barre latérale
    st.caption("Mode IA (Groq)" if use_llm else
               "Mode hors-ligne : réponses générées par des règles locales.")
    ctx = assistant.AirContext(city.name, level, pm10, pm25, PROFILES[profile_key],
                               peak_text, window_text)
    context_key = (city.name, profile_key)
    if st.session_state.get("chat_context") != context_key:
        st.session_state["chat"] = []
        st.session_state["chat_context"] = context_key
    st.session_state.setdefault("chat", [])
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    examples = ["Puis-je faire du sport aujourd'hui ?", "Quel masque choisir ?",
                "Pourquoi y a-t-il autant de poussière en harmattan ?"]
    cols = st.columns(len(examples))
    clicked = next((q for col, q in zip(cols, examples) if col.button(q)), None)
    question = st.chat_input("Ex. : mon enfant peut-il jouer dehors cet après-midi ?") or clicked
    if question:
        history_msgs = list(st.session_state["chat"])
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Réflexion…"):
                answer, src = assistant.ask(question, ctx, history_msgs, api_key=key_input or None)
            st.markdown(answer)
            st.caption(f"Réponse générée par : {'Groq' if src == 'llm' else 'règles locales'}")
            if src != "llm" and use_llm and assistant.last_error:
                # CHANGÉ : plus de détail technique brut affiché au jury
                st.warning("Le service IA est momentanément indisponible, réponse de secours affichée.")
        st.session_state["chat"].append({"role": "assistant", "content": answer})
    st.caption("L'assistant explique et oriente ; il ne remplace pas un professionnel de santé.")

# ---------------------------------------------------------------- ---------------------------------------------------------------- Partager
with t_share:
    st.header("Partager l'alerte avec vos proches")
    st.write("Un message court, prêt pour WhatsApp ou SMS, pour prévenir ceux qui n'ouvriront jamais ce tableau de bord.")
    lang = st.radio("Langue", ["fr", "en"], horizontal=True, format_func={"fr": "Français", "en": "English"}.get)
    target = st.selectbox("Pour qui ?", list(PROFILES), index=list(PROFILES).index(profile_key),
                          format_func=PROFILES.get, key="share_profile")
    msg = alerts.build_message(city.name, level, pm10, target, lang, best_today, peak_time)
    st.text_area("Message", msg, height=220)
    st.link_button("Envoyer par WhatsApp", alerts.whatsapp_link(msg))
    st.caption("Le lien ouvre WhatsApp avec le texte pré-rempli ; vous choisissez le destinataire.")

# ---------------------------------------------------------------- Historique
with t_hist:
    st.header(f"Tendances à {city.name}")
    if history is None or history.empty:
        st.info(f"Pas d'historique local : `python train.py --city {city.name} --download`.")
    else:
        period = st.selectbox("Période", ["3 derniers mois", "12 derniers mois", "Tout l'historique"], index=1)
        months = {"3 derniers mois": 3, "12 derniers mois": 12}.get(period)
        view = history.loc[history.index.max() - pd.DateOffset(months=months):] if months else history
        monthly, episodes = monthly_summary(view), detect_particle_episodes(view)
        total = int(monthly["observed_hours"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Heures mesurées", f"{total:,}".replace(",", " "))
        c2.metric("PM10 moyen", f"{view['pm10'].mean():.0f} µg/m³")
        c3.metric("Épisodes ≥ 3 h au-dessus de 100", len(episodes))

        st.subheader("Jours au-dessus du repère OMS 2021 (24 h)")
        who = risk.who_exceedance(view)
        w10 = who[who["pollutant"] == "pm10"]
        if not w10.empty:
            st.dataframe(w10.assign(share_over=(100 * w10["share_over"]).round(0))
                         .rename(columns={"year": "Année", "valid_days": "Jours valides", "days_over": "Jours > 45 µg/m³",
                                          "share_over": "% des jours", "annual_mean": "Moyenne PM10"})
                         [["Année", "Jours valides", "Jours > 45 µg/m³", "% des jours", "Moyenne PM10"]].round(1),
                         hide_index=True, use_container_width=True)
        season = risk.seasonal_contrast(view)
        if season:
            st.write(f"Moyenne PM10 en saison harmattan (nov.-mars) : **{season['harmattan_mean']:.0f}** µg/m³, "
                     f"hors saison : **{season['other_mean']:.0f}** (×{season['ratio']:.1f}).")

        st.subheader("Les heures les plus chargées")
        prof = risk.hourly_profile(view)
        if not prof.empty:
            st.plotly_chart(px.bar(x=prof.index, y=prof.values, labels={"x": "Heure", "y": "PM10 moyen (µg/m³)"}),
                            use_container_width=True)
        mat = risk.month_hour_matrix(view)
        if not mat.empty:
            heat = px.imshow(mat.values, x=list(range(24)), y=[MONTHS[m - 1][:4] for m in mat.index],
                             color_continuous_scale="YlOrRd", aspect="auto",
                             labels={"x": "Heure", "y": "Mois", "color": "PM10"})
            st.plotly_chart(heat, use_container_width=True)

        st.subheader("Concentration moyenne par mois")
        st.plotly_chart(px.line(monthly.reset_index(), x="time", y="pm10_mean", markers=True,
                                labels={"time": "Mois", "pm10_mean": "PM10 moyen (µg/m³)"}),
                        use_container_width=True)
        if not episodes.empty:
            ep = episodes.sort_values("start", ascending=False).head(8).copy()
            ep["start"], ep["end"] = ep["start"].dt.strftime("%d/%m/%Y %H:%M"), ep["end"].dt.strftime("%d/%m/%Y %H:%M")
            st.dataframe(ep.rename(columns={"start": "Début", "end": "Fin", "duration_hours": "Durée (h)",
                                            "peak_pm10": "Pic PM10", "mean_pm10": "PM10 moyen",
                                            "mean_dust": "Poussière moy."}).round(0),
                         hide_index=True, use_container_width=True)
        st.caption("Une concentration élevée n'identifie pas à elle seule l'origine saharienne.")

# ---------------------------------------------------------------- À propos
with t_about:
    st.header("Comment ça marche, et ce que ça vaut")
    import json
    metrics_file = paths["metrics"]
    if metrics_file.is_file():
        rep = json.loads(metrics_file.read_text(encoding="utf-8"))
        st.subheader("Évaluation honnête du modèle (bloc de test le plus récent)")
        rows = []
        for h, v in rep["horizons"].items():
            ex = v.get("exceedance_100", {})
            rows.append({"Horizon": f"+{h} h", "MAE modèle": round(v["model"]["mae"], 1),
                         "MAE persistance": round(v["baseline"]["mae"], 1),
                         "Gain": f"{100 * v.get('skill_vs_persistence', 0):+.0f} %",
                         "Couverture intervalle 80 %": f"{100 * v['coverage_80']:.0f} %",
                         "Brier dépassement": round(ex["brier_model"], 3) if ex.get("available") else None,
                         "Brier climatologie": round(ex["brier_climatology"], 3) if ex.get("available") else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Le gain sur la valeur exacte de PM10 est modeste ; la probabilité de dépasser 100 µg/m³ "
                   "est nettement meilleure qu'une simple moyenne climatique (Brier plus bas = mieux).")
        if rep.get("drivers_24h"):
            st.write("Variables qui pèsent le plus sur la prévision à 24 h : "
                     + ", ".join(f"`{d['feature']}`" for d in rep["drivers_24h"][:5]) + ".")
    st.markdown("""
- **Données** : Open-Meteo (réanalyse et prévision CAMS), un point de référence par ville.
- **IA** : prévision gradient boosting (+6/12/24 h) avec intervalles recalibrés (conformal), probabilité de dépassement,
  détection d'anomalies, assistant conversationnel Groq (facultatif).
- **Limites** : pas de capteur local ; la valeur n'indique pas l'origine de la poussière ; l'air intérieur n'est pas couvert ;
  conseils indicatifs, jamais un avis médical.
""")

st.caption(f"ClearAirDK · Track Sustainability · interface actualisée le {datetime.now().astimezone():%d/%m/%Y %H:%M}")
