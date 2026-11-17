import numpy as np
import pandas as pd
import pytest

from clearair import alerts, assistant, risk
from clearair import cities as CITY
from clearair.advice import advice_text, level_label, practical_tips
from clearair.insights import city_snapshot
from clearair.models import build_features, load_forecaster, predict, train_all
from tests.test_core import synthetic


def day_frame(values, start="2024-01-10 06:00"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"pm10": values, "pm2_5": np.array(values) * 0.3}, index=idx)


def test_who_exceedance_counts_days_over_limit_only_for_complete_days():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    frame = pd.DataFrame({"pm10": [60.0] * 24 + [30.0] * 24}, index=idx)
    out = risk.who_exceedance(frame)
    row = out[out["pollutant"] == "pm10"].iloc[0]
    assert row["valid_days"] == 2 and row["days_over"] == 1
    short = frame.iloc[:10]
    assert risk.who_exceedance(short).empty


def test_hourly_profile_and_matrix_shapes():
    frame = synthetic(40)
    assert len(risk.hourly_profile(frame)) == 24
    assert risk.month_hour_matrix(frame).shape == (12, 24)


def test_compare_schedule_finds_cleaner_window_and_reduction():
    values = [150, 140, 130, 120, 110, 100, 90, 40, 30, 30, 40, 80, 120]
    frame = day_frame(values, "2024-01-10 06:00")
    out = risk.compare_schedule(frame, duration=2, usual_start=9, activity="Marche, travail léger")
    assert out["best"]["start"].hour == 13 or out["best"]["start"].hour == 14
    assert out["reduction_percent"] > 0
    heavy = risk.exposure_index(frame.iloc[:2], "Travail physique, sport")
    rest = risk.exposure_index(frame.iloc[:2], "Repos / travail assis")
    assert heavy == pytest.approx(4 * rest)
    with pytest.raises(ValueError):
        risk.exposure_index(frame, "inconnue")


def test_best_window_respects_hour_bounds():
    frame = day_frame([200, 10, 10, 200, 200, 200], "2024-01-10 04:00")  # 04h-05h trop tôt
    out = risk.best_window_within(frame, 2, earliest=6, latest=19)
    assert out is None or out["start"].hour >= 6


def test_school_plan_traffic_lights():
    values = [10] * 4 + [60] * 3 + [300] * 3
    plan = risk.school_plan(day_frame(values, "2024-01-10 08:00"))
    assert set(plan["Feu"]) == {"vert", "orange", "rouge"}


def test_messages_fr_en_and_whatsapp_link():
    window = {"start": pd.Timestamp("2024-01-10 07:00"), "end": pd.Timestamp("2024-01-10 09:00")}
    fr = alerts.build_message("Dakar", 3, 130.0, "enfant", "fr", window, pd.Timestamp("2024-01-10 15:00"))
    en = alerts.build_message("Dakar", 3, 130.0, "enfant", "en", window)
    assert "Très élevé" in fr and "15:00" in fr and "07:00–09:00" in fr
    assert "Very high" in en and "children" in en
    assert alerts.whatsapp_link(fr).startswith("https://wa.me/?text=%F0%9F")
    assert "n/d" in alerts.build_message("Dakar", None, None, "grand_public")
    with pytest.raises(ValueError):
        alerts.build_message("Dakar", 1, 10, "enfant", "wo")


def test_every_profile_level_has_english_and_tips():
    from clearair.advice import PROFILES
    for p in PROFILES:
        for lvl in range(5):
            assert advice_text(p, lvl, "en")
    assert level_label(2, "en") == "High"
    assert any("FFP2" in t for t in practical_tips(4))
    assert practical_tips(None) == ()


class FakeClient:
    def __init__(self, text="Réponse test", fail=False):
        self.text, self.fail, self.calls = text, fail, []
        self.chat = self
        self.completions = self
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": text})()})()]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("réseau")
        return type("R", (), {"choices": self.choices})()


CTX = assistant.AirContext("Dakar", 3, 130.0, 40.0, "Enfant", "Pic vers 15:00.", "")


def test_assistant_uses_llm_with_context_and_safety_prompt():
    client = FakeClient()
    text, src = assistant.ask("Mon enfant peut-il jouer dehors ?", CTX, client=client)
    assert (text, src) == ("Réponse test", "llm")
    call = client.calls[0]
    assert "PM10 : 130" in call["messages"][-1]["content"]
    assert call["messages"][0]["role"] == "system"
    assert "pas médecin" in call["messages"][0]["content"]


def test_assistant_falls_back_on_error_and_never_sends_emergency_to_llm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    text, src = assistant.ask("masque ?", CTX, client=FakeClient(fail=True))
    assert src == "règles" and "FFP2" in text
    client = FakeClient()
    text, src = assistant.ask("J'ai une douleur à la poitrine", CTX, client=client)
    assert src == "règles" and "urgence" in text.lower() and not client.calls
    assert assistant.ask("sport ?", CTX)[1] == "règles"
    with pytest.raises(ValueError):
        assistant.ask("  ", CTX)


def test_groq_key_detection(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert not assistant.llm_available()
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert assistant.llm_available()


def test_city_snapshot_and_catalog():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    frame = pd.DataFrame({"pm10": [20] * 5 + [50, 200, 90, 60, 40], "pm2_5": [8] * 10}, index=idx)
    snap = city_snapshot(frame, idx[4] + pd.Timedelta(minutes=10))
    assert snap["pm10"] == 20 and snap["next24_max"] == 200
    assert snap["next24_peak_time"] == idx[6]
    assert len(CITY.CITIES) == 12 and CITY.get("Touba").slug == "touba"
    with pytest.raises(ValueError):
        CITY.get("Paris")


def test_model_has_calibrated_interval_and_exceedance_probability(tmp_path):
    frame = synthetic(120)
    report = train_all(frame, tmp_path / "m.joblib", tmp_path / "r.json", tmp_path / "a.joblib")
    h = report["horizons"]["12"]
    assert 0 <= h["coverage_80_raw"] <= 1 and "skill_vs_persistence" in h
    assert report["drivers_24h"]
    out = predict(load_forecaster(tmp_path / "m.joblib"), build_features(frame))
    for v in out.values():
        assert v["lower"] <= v["median"] + 1e-6 or v["lower"] <= v["upper"]
        assert v["p_exceed"] is None or 0 <= v["p_exceed"] <= 1


def test_paths_for_keeps_legacy_dakar_names():
    from clearair import config as C
    assert C.paths_for("dakar")["model"].name == "forecast.joblib"
    assert C.paths_for("touba")["csv"].name == "touba_hourly.csv"


def test_default_groq_model_is_not_the_retired_one():
    assert "llama-3.3-70b-versatile" not in assistant.DEFAULT_MODEL


def test_llm_error_is_exposed_for_the_interface(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    text, src = assistant.ask("masque ?", CTX, client=FakeClient(fail=True))
    assert src == "règles" and "réseau" in assistant.last_error
    assistant.ask("masque ?", CTX, client=FakeClient())
    assert assistant.last_error is None


def test_gpt_oss_gets_low_reasoning_effort():
    client = FakeClient()
    assistant.ask("sport ?", CTX, client=client, model="openai/gpt-oss-120b")
    assert client.calls[0]["reasoning_effort"] == "low" and client.calls[0]["max_tokens"] >= 800
    client2 = FakeClient()
    assistant.ask("sport ?", CTX, client=client2, model="autre/modele")
    assert "reasoning_effort" not in client2.calls[0]
