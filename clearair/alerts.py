"""Messages courts à partager (WhatsApp, SMS) : l'information là où les gens sont."""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd

from clearair.advice import advice_text, level_label, practical_tips

EMOJI = ("🟢", "🟡", "🟠", "🔴", "🟣")
_PROFILE_NAMES = {
    "fr": {"asthmatique": "personnes asthmatiques", "enfant": "enfants",
           "personne_agee": "personnes âgées", "travailleur_exterieur": "travailleurs en extérieur",
           "grand_public": "grand public"},
    "en": {"asthmatique": "people with asthma", "enfant": "children",
           "personne_agee": "older adults", "travailleur_exterieur": "outdoor workers",
           "grand_public": "general public"},
}


def build_message(city: str, level: int | None, pm10: float | None, profile: str,
                  lang: str = "fr", best_window: dict | None = None,
                  peak_time: pd.Timestamp | None = None) -> str:
    """Construit un bulletin partageable, avec rappel de ses limites."""
    if lang not in ("fr", "en"):
        raise ValueError("Langue non prise en charge : fr ou en.")
    fr = lang == "fr"
    icon = EMOJI[level] if level is not None else "⚪"
    title = f"{icon} ClearAirDK · {city}"
    value = f"PM10 : {pm10:.0f} µg/m³" if pm10 is not None else ("PM10 : n/d" if fr else "PM10: n/a")
    state = ("Air : " if fr else "Air: ") + level_label(level, lang)
    lines = [title, f"{state} ({value})",
             f"{'Pour' if fr else 'For'} {_PROFILE_NAMES[lang][profile]} : "
             f"{advice_text(profile, level, lang)}"]
    if peak_time is not None and pd.notna(peak_time):
        lines.append(("Pic de poussière prévu vers " if fr else "Dust peak expected around ")
                     + f"{peak_time:%H:%M}.")
    if best_window:
        lines.append(("Créneau le moins chargé : " if fr else "Least dusty slot: ")
                     + f"{best_window['start']:%H:%M}–{best_window['end']:%H:%M}.")
    tips = practical_tips(level, lang)
    if tips:
        lines.append("• " + tips[0])
    lines.append("Prototype indicatif, pas un avis médical." if fr
                 else "Indicative prototype, not medical advice.")
    return "\n".join(lines)


def whatsapp_link(message: str) -> str:
    """Lien wa.me pré-rempli ; l'utilisateur choisit lui-même le destinataire."""
    return "https://wa.me/?text=" + quote(message)
