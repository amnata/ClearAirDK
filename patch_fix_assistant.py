"""Applique 5 corrections à app.py et clearair/assistant.py (à lancer depuis le dossier ClearAirDK)."""
from pathlib import Path

def patch(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"ERREUR dans {path} : texte introuvable ou ambigu :\n{old[:80]}…")
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print(f"OK : {path}")

patch("app.py", [
    # 1) plus de créneau conseillé en pleine nuit : seulement entre 6 h et 19 h
    ("best_today = best_outdoor_window(future_p.head(48), 2) if not future_p.empty else None",
     "best_today = risk.best_window_within(future_p.head(48), 2, 6, 19) if not future_p.empty else None"),
    # 2) même règle dans l'onglet Planifier (sortie)
    ("            w = best_outdoor_window(day_fc, dur)",
     "            w = risk.best_window_within(day_fc, dur, 6, 19)"),
    # 3) tableau de prévision trié par horizon (6, 12, 24)
    ("        for h, v in ml.items():", "        for h, v in sorted(ml.items()):"),
    # 4) la conversation repart de zéro quand la ville ou le profil change
    ('    st.session_state.setdefault("chat", [])',
     '    context_key = (city.name, profile_key)\n'
     '    if st.session_state.get("chat_context") != context_key:\n'
     '        st.session_state["chat"] = []\n'
     '        st.session_state["chat_context"] = context_key\n'
     '    st.session_state.setdefault("chat", [])'),
])

patch("clearair/assistant.py", [
    # 5) consignes plus strictes pour l'assistant
    ('    "(point de référence urbain, prévision indicative) quand c\'est utile."\n)',
     '    "(point de référence urbain, prévision indicative) quand c\'est utile. "\n'
     '    "Vouvoie toujours la personne. Utilise uniquement le profil et la ville du contexte, "\n'
     '    "sans en inventer d\'autres (par exemple asthme si le profil est Enfant). "\n'
     '    "Ne recommande jamais une activité entre 19 h et 6 h : propose uniquement le créneau "\n'
     '    "du contexte, ou dis qu\'aucun créneau n\'est favorable aujourd\'hui."\n)'),
])
print("Terminé. Lancez : python -m pytest -q")