"""Assistant conversationnel : Groq si une clé est fournie, sinon règles locales.

L'assistant ne reçoit que le contexte d'air du jour et le profil choisi. Il explique et conseille ; il ne diagnostique pas.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

from clearair.advice import advice_text, level_label, practical_tips

# llama-3.3-70b-versatile a été retiré par Groq le 16 août 2026.
last_error: str | None = None  # dernière erreur du service LLM (affichée par l'interface)
DEFAULT_MODEL = os.environ.get("CLEARAIR_LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "Tu es l'assistant de ClearAirDK, un outil de sensibilisation à la poussière et à la "
    "qualité de l'air au Sénégal. Réponds dans la langue de la question (français par "
    "défaut), en phrases courtes et simples, sans jargon, en 120 mots maximum. "
    "Appuie-toi UNIQUEMENT sur le contexte fourni pour les chiffres ; si une donnée manque, "
    "dis-le. Tu n'es pas médecin : ne pose aucun diagnostic, ne donne aucune posologie, "
    "et recommande de consulter un professionnel de santé ou de se rendre aux urgences en "
    "cas de gêne respiratoire importante, de douleur thoracique ou de malaise. "
    "Ne prétends pas connaître l'origine exacte de la poussière. Rappelle les limites "
    "(point de référence urbain, prévision indicative) quand c'est utile."
)


@dataclass
class AirContext:
    city: str
    level: int | None
    pm10: float | None
    pm2_5: float | None
    profile: str
    peak_text: str = ""
    window_text: str = ""

    def as_text(self) -> str:
        pm10 = f"{self.pm10:.0f} µg/m³" if self.pm10 is not None else "inconnu"
        pm25 = f"{self.pm2_5:.0f} µg/m³" if self.pm2_5 is not None else "inconnu"
        lines = [f"Ville : {self.city}", f"Niveau indicatif : {level_label(self.level)}",
                 f"PM10 : {pm10} ; PM2.5 : {pm25}", f"Profil : {self.profile}"]
        if self.peak_text:
            lines.append(self.peak_text)
        if self.window_text:
            lines.append(self.window_text)
        return "\n".join(lines)


EMERGENCY = ("poitrine", "etouff", "suffoqu", "malaise", "ne peux plus respirer",
             "ne respire plus", "bleu", "chest pain", "can't breathe", "cannot breathe")


def _norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def offline_answer(question: str, ctx: AirContext) -> str:
    """Réponse sans LLM : mots-clés + conseils validés du projet. Volontairement prudente."""
    q = _norm(question)
    if any(word in q for word in EMERGENCY):
        return ("⚠️ Ce que vous décrivez peut être urgent. Ne comptez pas sur cet outil : "
                "contactez immédiatement un service d'urgence ou un professionnel de santé.")
    state = f"À {ctx.city}, l'air est actuellement de niveau « {level_label(ctx.level)} »"
    if ctx.pm10 is not None:
        state += f" (PM10 ≈ {ctx.pm10:.0f} µg/m³)"
    state += "."
    tips = practical_tips(ctx.level)
    if re.search(r"masque|mask", q):
        body = next((t for t in practical_tips(3) if "FFP2" in t), tips[0] if tips else "")
    elif re.search(r"fenetre|aer|maison|chez moi|interieur", q):
        body = " ".join(practical_tips(2)[:2])
    elif re.search(r"sport|courir|foot|jogging|exercice|sortir|sortie|marche", q):
        body = advice_text("grand_public", ctx.level)
        if ctx.window_text:
            body += " " + ctx.window_text
    elif re.search(r"enfant|ecole|bebe|fils|fille", q):
        body = advice_text("enfant", ctx.level)
    elif re.search(r"asthm|toux|respir|souffle", q):
        body = advice_text("asthmatique", ctx.level) + " En cas de gêne importante, consultez."
    elif re.search(r"harmattan|poussiere|sahara|pourquoi", q):
        body = ("Pendant l'harmattan (environ novembre à mars), un vent sec peut transporter "
                "de la poussière. Une valeur élevée seule ne prouve pas son origine.")
    else:
        body = advice_text(ctx.profile, ctx.level)
    return f"{state} {body}\n\n(Mode hors-ligne : réponses prudentes à base de règles.)"


def llm_available(api_key: str | None = None) -> bool:
    return bool(api_key or os.environ.get("GROQ_API_KEY"))


def ask(question: str, ctx: AirContext, history: list[dict] | None = None,
        api_key: str | None = None, model: str = DEFAULT_MODEL, client=None) -> tuple[str, str]:
    """Retourne (réponse, source) où source vaut « llm » ou « règles »."""
    global last_error
    last_error = None
    question = question.strip()
    if not question:
        raise ValueError("La question est vide.")
    if any(word in _norm(question) for word in EMERGENCY):
        return offline_answer(question, ctx), "règles"  # l'urgence ne passe jamais par le LLM
    key = api_key or os.environ.get("GROQ_API_KEY")
    if client is None and not key:
        return offline_answer(question, ctx), "règles"
    try:
        if client is None:
            from groq import Groq
            client = Groq(api_key=key, timeout=20.0)
        messages = [m for m in (history or [])[-6:] if m.get("role") in ("user", "assistant")]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        messages.append({"role": "user",
                         "content": f"Contexte (mesures et prévisions) :\n{ctx.as_text()}\n\n"
                                    f"Question : {question}"})
        options = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        response = client.chat.completions.create(model=model, max_tokens=1000,
                                                  messages=messages, **options)
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise ValueError("réponse vide")
        return text.strip(), "llm"
    except Exception as exc:  # réseau, quota, clé ou modèle invalide : repli sur les règles
        last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        return offline_answer(question, ctx), "règles"
