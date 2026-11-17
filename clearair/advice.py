"""Niveaux indicatifs et recommandations adaptées aux profils."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AirLevel:
    code: int
    label: str
    color: str


LEVELS = (
    AirLevel(0, "Faible", "#2e8b57"),
    AirLevel(1, "Modéré", "#d4a017"),
    AirLevel(2, "Élevé", "#e87524"),
    AirLevel(3, "Très élevé", "#c83232"),
    AirLevel(4, "Extrême", "#742b8a"),
)

PROFILES = {
    "asthmatique": "Personne asthmatique",
    "enfant": "Enfant",
    "personne_agee": "Personne âgée",
    "travailleur_exterieur": "Travailleur en extérieur",
    "grand_public": "Grand public",
}

_ADVICE = {
    "asthmatique": (
        "Activités habituelles possibles. Gardez votre traitement prescrit accessible.",
        "Surveillez les symptômes et réduisez l'effort prolongé dehors.",
        "Limitez l'effort extérieur ; privilégiez un espace intérieur moins poussiéreux.",
        "Évitez l'activité physique dehors. Suivez votre plan d'action médical.",
        "Restez dans un espace intérieur protégé si possible. En cas de gêne respiratoire importante, demandez une aide médicale.",
    ),
    "enfant": (
        "Jeux extérieurs habituels possibles.",
        "Évitez les efforts très intenses et surveillez l'apparition de symptômes.",
        "Réduisez les activités longues en plein air ; faites des pauses à l'intérieur.",
        "Privilégiez les activités à l'intérieur et surveillez la respiration.",
        "Gardez l'enfant dans un espace intérieur protégé si possible ; demandez conseil à un professionnel en cas de symptômes.",
    ),
    "personne_agee": (
        "Activités habituelles possibles.",
        "Évitez les efforts intenses si vous êtes sensible à la poussière.",
        "Réduisez les sorties et les efforts prolongés en extérieur.",
        "Limitez les activités dehors et privilégiez un lieu intérieur moins poussiéreux.",
        "Restez à l'intérieur si possible ; sollicitez une aide médicale en cas de symptômes préoccupants.",
    ),
    "travailleur_exterieur": (
        "Activités extérieures habituelles possibles.",
        "Faites des pauses et évitez les efforts excessifs aux heures les plus poussiéreuses.",
        "Réduisez si possible la durée des tâches exposées et prévoyez des pauses à l'abri.",
        "Réorganisez ou reportez les tâches non essentielles ; suivez les consignes de sécurité au travail.",
        "Évitez l'exposition non indispensable et suivez les consignes locales de santé et sécurité.",
    ),
    "grand_public": (
        "Activités habituelles possibles.",
        "Les personnes sensibles peuvent réduire les efforts prolongés en extérieur.",
        "Réduisez les efforts physiques prolongés dehors.",
        "Privilégiez les activités à l'intérieur et limitez l'exposition à la poussière.",
        "Réduisez au minimum l'exposition extérieure et suivez les consignes sanitaires locales.",
    ),
}


def _level_for(value: float | None, thresholds: tuple[float, ...]) -> int | None:
    if value is None or value != value:
        return None
    for level, threshold in enumerate(thresholds):
        if value < threshold:
            return level
    return len(thresholds)


def level_from(pm10: float | None, pm25: float | None = None) -> int | None:
    """Retourne un niveau indicatif selon des repères distincts PM10/PM2.5."""
    levels = [
        level for level in (
            _level_for(pm10, (20, 45, 100, 250)),
            _level_for(pm25, (10, 25, 50, 100)),
        ) if level is not None
    ]
    return max(levels) if levels else None


def advice_for(profile: str, level: int | None) -> str:
    """Retourne une recommandation en français, y compris si le niveau manque."""
    if profile not in _ADVICE:
        raise ValueError(f"Profil non reconnu : {profile}")
    if level is None:
        return "Données insuffisantes pour évaluer la qualité de l'air. Consultez les sources locales."
    if not 0 <= level < len(LEVELS):
        raise ValueError(f"Niveau non reconnu : {level}")
    return _ADVICE[profile][level]


# --- Version anglaise et gestes pratiques -------------------------------------

LEVEL_LABELS_EN = ("Low", "Moderate", "High", "Very high", "Extreme")

_ADVICE_EN = {
    "asthmatique": (
        "Usual activities are fine. Keep your prescribed reliever within reach.",
        "Watch for symptoms and avoid long effort outdoors.",
        "Limit outdoor effort; prefer a less dusty indoor space.",
        "Avoid outdoor exercise. Follow your medical action plan.",
        "Stay in a protected indoor space if possible. Seek medical help if breathing becomes difficult.",
    ),
    "enfant": (
        "Usual outdoor play is fine.",
        "Avoid very intense effort and watch for symptoms.",
        "Cut long outdoor activities; take breaks indoors.",
        "Prefer indoor activities and watch the child's breathing.",
        "Keep the child indoors if possible; ask a professional for advice if symptoms appear.",
    ),
    "personne_agee": (
        "Usual activities are fine.",
        "Avoid intense effort if you are sensitive to dust.",
        "Reduce outings and long outdoor effort.",
        "Limit outdoor activity and prefer a less dusty indoor place.",
        "Stay indoors if possible; seek medical help for worrying symptoms.",
    ),
    "travailleur_exterieur": (
        "Usual outdoor work is fine.",
        "Take breaks and avoid heavy effort at the dustiest hours.",
        "Shorten exposed tasks if you can and plan breaks in shelter.",
        "Reschedule non-essential tasks; follow workplace safety rules.",
        "Avoid non-essential exposure and follow local health and safety guidance.",
    ),
    "grand_public": (
        "Usual activities are fine.",
        "Sensitive people may reduce long outdoor effort.",
        "Reduce long physical effort outdoors.",
        "Prefer indoor activities and limit dust exposure.",
        "Keep outdoor exposure to a minimum and follow local health guidance.",
    ),
}

_TIPS = {
    "fr": {
        "low": ("Aérez la maison aux heures les moins poussiéreuses.",),
        "mid": (
            "Fermez les fenêtres pendant les pics de poussière, aérez en dehors.",
            "Nettoyez avec un chiffon humide plutôt qu'en balayant à sec.",
            "Buvez régulièrement ; la poussière dessèche gorge et yeux.",
        ),
        "high": (
            "Si vous devez sortir : un masque FFP2/N95 bien ajusté filtre les particules ; "
            "un tissu ordinaire ou un masque chirurgical protège peu contre la poussière fine.",
            "Fermez portes et fenêtres, arrosez ou humidifiez le sol devant chez vous.",
            "Ne brûlez pas de déchets : cela ajoute des particules.",
            "Gêne respiratoire, sifflements ou douleur thoracique : consultez rapidement.",
        ),
    },
    "en": {
        "low": ("Ventilate your home at the least dusty hours.",),
        "mid": (
            "Close windows during dust peaks and ventilate outside them.",
            "Clean with a damp cloth rather than dry sweeping.",
            "Drink regularly; dust dries the throat and eyes.",
        ),
        "high": (
            "If you must go out, a well-fitted FFP2/N95 mask filters particles; "
            "cloth or surgical masks protect little against fine dust.",
            "Close doors and windows and dampen the ground in front of your home.",
            "Do not burn waste: it adds particles.",
            "Breathing trouble, wheezing or chest pain: seek care promptly.",
        ),
    },
}


def advice_text(profile: str, level: int | None, lang: str = "fr") -> str:
    """Conseil par profil en français ou anglais."""
    if lang == "fr":
        return advice_for(profile, level)
    if profile not in _ADVICE_EN:
        raise ValueError(f"Profil non reconnu : {profile}")
    if level is None:
        return "Not enough data to assess air quality. Check local sources."
    if not 0 <= level < len(LEVELS):
        raise ValueError(f"Niveau non reconnu : {level}")
    return _ADVICE_EN[profile][level]


def level_label(level: int | None, lang: str = "fr") -> str:
    if level is None:
        return "Inconnu" if lang == "fr" else "Unknown"
    return LEVELS[level].label if lang == "fr" else LEVEL_LABELS_EN[level]


def practical_tips(level: int | None, lang: str = "fr") -> tuple[str, ...]:
    """Gestes simples pour réduire l'exposition, selon le niveau."""
    if level is None:
        return ()
    group = "low" if level == 0 else "mid" if level <= 2 else "high"
    return _TIPS[lang][group]
