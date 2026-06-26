"""
Phase 1 Wound/Tissue Recovery — Streamlit Demo Standalone v014
================================================================

One-file doctor-facing demo for the three Phase-1 layers only:
  Layer 1: Healing Phase Estimator (HMM-style interpretable phase posterior)
  Layer 2: Deviation Detector (trajectory bands + clinical deviation rules)
  Layer 3: Nutritional Gap Engine (rule-based phase-specific target scoring)

No external project modules are required.
No Phase-2 logic is included.

Run:
  streamlit run phase1_wound_recovery_streamlit_demo_standalone_014.py
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "phase1_streamlit_demo_standalone_v014"
IST = timezone(timedelta(hours=5, minutes=30))
PHASES = ["haemostasis", "inflammation", "proliferation", "remodelling"]
SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}

st.set_page_config(page_title="Phase 1 Wound Recovery Demo", page_icon="🩹", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE TABLES
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_TIMELINE = [
    {"phase": "haemostasis", "window": "0–3 hours", "purpose": "Clot formation and bleeding control"},
    {"phase": "inflammation", "window": "0–5 days", "purpose": "Immune cleanup, CRP/WBC activity"},
    {"phase": "proliferation", "window": "5–21 days", "purpose": "Fibroblast activity, collagen synthesis, granulation"},
    {"phase": "remodelling", "window": "21+ days", "purpose": "Collagen maturation and tensile strength"},
]

PHASE_REFERENCE_ROWS = [
    {
        "Phase": "Haemostasis",
        "Duration": "0–3 hours",
        "Key biology": "Platelet aggregation and fibrin clot formation",
        "Critical nutrition": "Vitamin K, hydration, electrolyte balance",
        "Primary monitoring": "Vitals stabilisation and blood loss estimate",
    },
    {
        "Phase": "Inflammation",
        "Duration": "0–5 days",
        "Key biology": "Neutrophil/macrophage recruitment and debris clearance",
        "Critical nutrition": "Zinc, Vitamin C, protein 1.5–2.0 g/kg/day",
        "Primary monitoring": "CRP rising then plateauing, WBC, temperature, HR",
    },
    {
        "Phase": "Proliferation",
        "Duration": "5–21 days",
        "Key biology": "Fibroblast migration, collagen synthesis, angiogenesis",
        "Critical nutrition": "Protein 1.8–2.0 g/kg/day, arginine, Vitamin C 500–1000 mg, caloric surplus",
        "Primary monitoring": "CRP declining, prealbumin rising, albumin stabilising",
    },
    {
        "Phase": "Remodelling",
        "Duration": "21+ days",
        "Key biology": "Collagen cross-linking, scar maturation, tensile strength gain",
        "Critical nutrition": "Protein maintenance 1.2–1.5 g/kg/day, ongoing Vitamin C",
        "Primary monitoring": "Labs normalising and pain reducing",
    },
]

MONITORING_SIGNAL_ROWS = [
    {"Signal": "Heart rate", "Source": "Wearable", "What it reveals": "Infection onset, pain, haemodynamic stress", "Cannot reveal": "Healing rate or tissue repair quality", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "Temperature", "Source": "Wearable", "What it reveals": "Systemic infection/inflammatory state", "Cannot reveal": "Local wound temperature", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "SpO₂", "Source": "Wearable", "What it reveals": "Systemic oxygenation", "Cannot reveal": "Wound-bed oxygenation specifically", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "CRP", "Source": "Lab", "What it reveals": "Inflammation trajectory", "Cannot reveal": "Cause of inflammation", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "WBC", "Source": "Lab", "What it reveals": "Infection versus immune activity", "Cannot reveal": "Tissue repair progress directly", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "Albumin", "Source": "Lab", "What it reveals": "Protein synthesis capacity/nutrition status", "Cannot reveal": "Immediate nutrition changes", "Layer use": "Layer 1 + Layer 2"},
    {"Signal": "Prealbumin", "Source": "Lab", "What it reveals": "Faster protein-status response", "Cannot reveal": "Direct wound-bed status", "Layer use": "Layer 1 + Layer 2 + Layer 3 interaction"},
    {"Signal": "Glucose", "Source": "Lab", "What it reveals": "Glycaemic control", "Cannot reveal": "Direct healing rate", "Layer use": "Layer 1 + Layer 2 + Layer 3 interaction"},
    {"Signal": "Zinc", "Source": "Lab", "What it reveals": "Zinc sufficiency for epithelialisation", "Cannot reveal": "Cellular zinc at wound site", "Layer use": "Layer 3 context"},
    {"Signal": "Pain/appetite/nausea", "Source": "Self-report", "What it reveals": "Tolerance and complication signals", "Cannot reveal": "Objective wound closure", "Layer use": "Layer 1 + Layer 3 context"},
]

HMM_COMPARISON_ROWS = [
    {"Algorithm": "Hidden Markov Model", "Verdict": "Chosen", "Reason": "Purpose-built for hidden phase inference with interpretable transition/emission probabilities."},
    {"Algorithm": "LSTM / RNN", "Verdict": "Rejected", "Reason": "Needs large training data and is harder for physicians to interpret."},
    {"Algorithm": "Logistic regression", "Verdict": "Rejected", "Reason": "Cannot model temporal dependencies between observations."},
    {"Algorithm": "Rule-based phase rules", "Verdict": "Rejected", "Reason": "Too rigid for delayed transitions in diabetic or complex surgical patients."},
]

DEVIATION_RULE_ROWS = [
    {"Trigger": "CRP plateau / re-elevation after Day 3", "Slider to test": "CRP", "Layer 2 flag": "inflammation_not_resolving"},
    {"Trigger": "Inflammation posterior remains high after Day 7", "Slider to test": "CRP, temperature, WBC, day", "Layer 2 flag": "phase_transition_delay"},
    {"Trigger": "SpO₂ below expected range during proliferation", "Slider to test": "SpO₂", "Layer 2 flag": "oxygenation_concern"},
    {"Trigger": "Prealbumin low / flat / declining from Day 5", "Slider to test": "Prealbumin", "Layer 2 flag": "protein_synthesis_impairment"},
    {"Trigger": "Glucose > 180 mg/dL", "Slider to test": "Glucose", "Layer 2 flag": "glucose_dysregulation"},
    {"Trigger": "Heart rate > 100 after Day 2", "Slider to test": "Heart rate", "Layer 2 flag": "tachycardia_review"},
    {"Trigger": "Temperature ≥ 38.5°C after Day 2", "Slider to test": "Temperature", "Layer 2 flag": "fever_pattern"},
    {"Trigger": "Serum zinc < 60", "Slider to test": "Serum zinc", "Layer 2 flag": "zinc_deficiency_context"},
]

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "haemostasis_walkthrough": {
        "label": "1. Haemostasis walkthrough",
        "expected_phase": "haemostasis",
        "description": "Immediate post-op example. Focus: clot formation window, vitals stabilisation, blood loss visibility, hydration/electrolyte monitoring.",
        "patient_id": "P0001", "age": 49, "sex": "female", "surgery_type": "abdominal", "asa_score": 2, "bmi": 27.1,
        "weight_kg": 70.0, "diabetes_status": False, "nrs_2002_score": 1, "day_post_op": 0,
        "vitals": {"heart_rate": 92, "temperature": 36.8, "spo2": 98},
        "labs": {"CRP": 8, "WBC": 8800, "albumin": 3.8, "prealbumin": 170, "glucose": 116, "zinc": 80},
        "nutrition": {"calories_kcal": 0, "protein_g": 0, "carbohydrates_g": 0, "vitamin_c_mg": 0, "hydration_ml": 1200},
        "self_report": {"pain_score": 5, "appetite": "not_started", "nausea": False},
    },
    "inflammation_walkthrough": {
        "label": "2. Inflammation walkthrough",
        "expected_phase": "inflammation",
        "description": "Day 3 example. CRP/WBC and mild temperature/HR activity make the phase posterior favour inflammation.",
        "patient_id": "P0101", "age": 44, "sex": "female", "surgery_type": "abdominal", "asa_score": 1, "bmi": 24.8,
        "weight_kg": 68.0, "diabetes_status": False, "nrs_2002_score": 1, "day_post_op": 3,
        "vitals": {"heart_rate": 101, "temperature": 38.1, "spo2": 98},
        "labs": {"CRP": 92, "WBC": 12400, "albumin": 3.6, "prealbumin": 142, "glucose": 124, "zinc": 82},
        "nutrition": {"calories_kcal": 1904, "protein_g": 102, "carbohydrates_g": 220, "vitamin_c_mg": 320, "hydration_ml": 2244},
        "self_report": {"pain_score": 4, "appetite": "fair", "nausea": False},
    },
    "proliferation_walkthrough": {
        "label": "3. Proliferation walkthrough",
        "expected_phase": "proliferation",
        "description": "Day 8 example. CRP/WBC are declining and prealbumin is rising, then Layer 3 checks high-demand collagen-synthesis nutrition targets.",
        "patient_id": "P0042", "age": 58, "sex": "unknown", "surgery_type": "abdominal", "asa_score": 2, "bmi": 31.0,
        "weight_kg": 82.0, "diabetes_status": True, "nrs_2002_score": 3, "day_post_op": 8,
        "vitals": {"heart_rate": 86, "temperature": 37.1, "spo2": 97},
        "labs": {"CRP": 38, "WBC": 8900, "albumin": 3.3, "prealbumin": 156, "glucose": 142, "zinc": 72},
        "nutrition": {"calories_kcal": 1886, "protein_g": 90.2, "carbohydrates_g": 300, "vitamin_c_mg": 60, "hydration_ml": 2296},
        "self_report": {"pain_score": 3, "appetite": "reduced", "nausea": False},
    },
    "remodelling_walkthrough": {
        "label": "4. Remodelling walkthrough",
        "expected_phase": "remodelling",
        "description": "Day 24 example. Labs and vitals are normalising, pain is low, and Layer 1 favours remodelling.",
        "patient_id": "P0300", "age": 51, "sex": "male", "surgery_type": "abdominal", "asa_score": 2, "bmi": 26.2,
        "weight_kg": 72.0, "diabetes_status": False, "nrs_2002_score": 1, "day_post_op": 24,
        "vitals": {"heart_rate": 78, "temperature": 36.9, "spo2": 98},
        "labs": {"CRP": 7, "WBC": 7200, "albumin": 3.7, "prealbumin": 168, "glucose": 104, "zinc": 78},
        "nutrition": {"calories_kcal": 2016, "protein_g": 100.8, "carbohydrates_g": 230, "vitamin_c_mg": 220, "hydration_ml": 2304},
        "self_report": {"pain_score": 1, "appetite": "good", "nausea": False},
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def priority_icon(priority: Optional[str]) -> str:
    return {"CRITICAL": "🔴", "HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟡", "NONE": "🟢", None: "⚪"}.get(str(priority).upper(), "⚪")


def phase_icon(phase: str) -> str:
    return {"haemostasis": "🩸", "inflammation": "🔥", "proliferation": "🧱", "remodelling": "🧬"}.get(str(phase).lower(), "⚪")


def clean_float(value: Any, ndigits: int = 2) -> Any:
    try:
        if value is None:
            return None
        return round(float(value), ndigits)
    except Exception:
        return value


def format_post_op_day(value: Any) -> str:
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def _normalize(vals: Sequence[float]) -> List[float]:
    total = sum(max(float(v), 0.0) for v in vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [max(float(v), 0.0) / total for v in vals]


def _latest_values(history: Mapping[str, List[Dict[str, Any]]], signal: str, n: int = 2) -> List[float]:
    rows = sorted(history.get(signal, []) or [], key=lambda r: float(r.get("day_post_op", 0)))
    return [float(r.get("value", 0)) for r in rows[-n:]]


def _latest_times(history: Mapping[str, List[Dict[str, Any]]], signal: str, n: int = 2) -> List[float]:
    rows = sorted(history.get(signal, []) or [], key=lambda r: float(r.get("day_post_op", 0)))
    return [float(r.get("day_post_op", 0)) for r in rows[-n:]]

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PATIENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

def make_history(day: int, labs: Mapping[str, Any], vitals: Mapping[str, Any], scenario_key: str) -> Dict[str, List[Dict[str, Any]]]:
    day = int(round(float(day)))

    def t(d: float) -> str:
        base = datetime(2026, 6, 14, 8, 0, tzinfo=IST)
        return (base + timedelta(days=float(d))).isoformat()

    current = {
        "CRP": float(labs.get("CRP", 50)), "WBC": float(labs.get("WBC", 9000)),
        "albumin": float(labs.get("albumin", 3.4)), "prealbumin": float(labs.get("prealbumin", 130)),
        "glucose": float(labs.get("glucose", 120)), "spo2": float(vitals.get("spo2", 97)),
        "heart_rate": float(vitals.get("heart_rate", 85)), "temperature": float(vitals.get("temperature", 37.1)),
    }

    if scenario_key == "haemostasis_walkthrough":
        raw = {s: [(0, v)] for s, v in current.items()}
    elif scenario_key == "inflammation_walkthrough":
        raw = {
            "CRP": [(1, 45), (2, 72), (day, current["CRP"])], "WBC": [(1, 9800), (2, 11200), (day, current["WBC"])],
            "albumin": [(1, 3.7), (day, current["albumin"])], "prealbumin": [(2, 150), (day, current["prealbumin"])],
            "glucose": [(2, 126), (day, current["glucose"])], "spo2": [(1, 98), (2, 98), (day, current["spo2"])],
            "heart_rate": [(1, 94), (2, 98), (day, current["heart_rate"])], "temperature": [(1, 37.3), (2, 37.8), (day, current["temperature"])],
        }
    elif scenario_key == "remodelling_walkthrough":
        raw = {
            "CRP": [(5, 48), (12, 18), (day, current["CRP"])], "WBC": [(5, 9800), (12, 8200), (day, current["WBC"])],
            "albumin": [(7, 3.3), (14, 3.5), (day, current["albumin"])], "prealbumin": [(7, 122), (14, 151), (day, current["prealbumin"])],
            "glucose": [(20, 112), (23, 108), (day, current["glucose"])], "spo2": [(20, 98), (23, 98), (day, current["spo2"])],
            "heart_rate": [(20, 82), (23, 80), (day, current["heart_rate"])], "temperature": [(20, 37.0), (23, 36.9), (day, current["temperature"])],
        }
    else:
        raw = {
            "CRP": [(2, 118), (5, 84), (day, current["CRP"])], "WBC": [(2, 13500), (5, 11200), (day, current["WBC"])],
            "albumin": [(3, 3.2), (day, current["albumin"])], "prealbumin": [(5, 122), (day, current["prealbumin"])],
            "glucose": [(max(1, day - 1), 138), (day, current["glucose"])], "spo2": [(max(1, day - 2), 97), (max(1, day - 1), 97), (day, current["spo2"])],
            "heart_rate": [(max(1, day - 2), 89), (max(1, day - 1), 87), (day, current["heart_rate"])],
            "temperature": [(max(1, day - 2), 37.0), (max(1, day - 1), 37.0), (day, current["temperature"])],
        }

    out: Dict[str, List[Dict[str, Any]]] = {}
    for signal, points in raw.items():
        seen = set()
        rows = []
        for d, v in points:
            dd = float(d)
            if dd in seen:
                continue
            seen.add(dd)
            rows.append({"timestamp": t(dd), "day_post_op": int(dd) if dd.is_integer() else dd, "value": float(v)})
        out[signal] = sorted(rows, key=lambda r: float(r["day_post_op"]))
    return out


def _latest_two(history: Mapping[str, List[Dict[str, Any]]], signal: str) -> Tuple[Optional[float], Optional[float]]:
    vals = _latest_values(history, signal, 2)
    if not vals:
        return None, None
    if len(vals) == 1:
        return None, vals[0]
    return vals[-2], vals[-1]


def infer_observed_signals(day: int, history: Mapping[str, List[Dict[str, Any]]], vitals: Mapping[str, Any], labs: Mapping[str, Any], self_report: Mapping[str, Any]) -> Dict[str, str]:
    hr, temp, spo2 = float(vitals.get("heart_rate", 85)), float(vitals.get("temperature", 37.1)), float(vitals.get("spo2", 97))
    glucose, albumin = float(labs.get("glucose", 120)), float(labs.get("albumin", 3.4))
    pain = float(self_report.get("pain_score", 3))

    def trend_category(signal: str, up: float, down: float, normal_low: Optional[float] = None) -> str:
        prev, curr = _latest_two(history, signal)
        if curr is None:
            return "unknown"
        if normal_low is not None and curr < normal_low:
            return "normal_low"
        if prev is None:
            return "elevated_unknown_direction"
        delta = curr - prev
        if delta > up:
            return "rising"
        if delta < -down:
            return "declining"
        return "plateau"

    crp = trend_category("CRP", up=8, down=8, normal_low=10)
    wbc = trend_category("WBC", up=700, down=700)
    if wbc == "elevated_unknown_direction":
        wbc = "elevated" if float(labs.get("WBC", 9000)) > 11000 else "normal"

    prev_pre, curr_pre = _latest_two(history, "prealbumin")
    if curr_pre is None:
        prealb = "unknown"
    elif prev_pre is not None and curr_pre - prev_pre > 8:
        prealb = "rising"
    elif prev_pre is not None and curr_pre - prev_pre <= 2:
        prealb = "flat_declining"
    elif curr_pre < 150:
        prealb = "low"
    else:
        prealb = "normal_stable"

    if albumin < 3.5:
        alb = "low"
    else:
        alb = "normal_stable"

    if pain >= 7:
        pain_cat = "high_current_pain_unknown_trend"
    elif 5 <= day < 21 and pain <= 3:
        pain_cat = "improving"
    elif day >= 21 and pain <= 2:
        pain_cat = "low_current_pain_unknown_trend"
    else:
        pain_cat = "stable"

    return {
        "HR_trend": "sustained_tachycardia" if hr >= 110 else ("elevated" if hr >= 100 else "normal"),
        "Temp_pattern": "sustained_fever" if temp >= 38.5 else ("low_grade" if temp >= 38.0 else "afebrile"),
        "SpO2_level": "low" if spo2 < 93 else ("borderline" if spo2 < 95 else "normal"),
        "CRP_velocity": crp,
        "WBC_pattern": wbc,
        "Albumin_trend": alb,
        "Prealbumin_trend": prealb,
        "Glucose_level": "severely_elevated" if glucose >= 250 else ("elevated" if glucose > 180 else "controlled"),
        "Pain_trajectory": pain_cat,
    }


def build_phase1_state(demo: Mapping[str, Any], overrides: Mapping[str, Any], scenario_key: str) -> Dict[str, Any]:
    patient_id = str(overrides.get("patient_id") or demo["patient_id"])
    day = int(overrides.get("day_post_op", demo["day_post_op"]))
    weight = float(overrides.get("weight_kg", demo["weight_kg"]))
    age = int(overrides.get("age", demo["age"]))
    bmi = float(overrides.get("bmi", demo["bmi"]))
    vitals = {**dict(demo["vitals"]), **overrides.get("vitals", {})}
    labs = {**dict(demo["labs"]), **overrides.get("labs", {})}
    nutrition = {**dict(demo["nutrition"]), **overrides.get("nutrition", {})}
    self_report = overrides.get("self_report", demo.get("self_report", {}))

    surgery_dt = datetime(2026, 6, 14, 8, 0, tzinfo=IST)
    assessment_dt = surgery_dt + timedelta(days=day)
    history = make_history(day, labs, vitals, scenario_key)
    observed = infer_observed_signals(day, history, vitals, labs, self_report)

    phase1_input = {
        "surgery_datetime": surgery_dt.isoformat(),
        "assessment_datetime": assessment_dt.isoformat(),
        "day_post_op": int(day),
        "observed_signals": observed,
        "clinical_profile": {
            "surgery_type": overrides.get("surgery_type", demo["surgery_type"]),
            "asa_score": int(overrides.get("asa_score", demo["asa_score"])),
            "bmi": bmi,
            "age": age,
            "diabetes_status": bool(overrides.get("diabetes_status", demo["diabetes_status"])),
            "nrs_2002_score": float(overrides.get("nrs_2002_score", demo["nrs_2002_score"])),
            "immunosuppression": bool(overrides.get("immunosuppression", False)),
            "smoking_status": bool(overrides.get("smoking_status", False)),
            "body_weight_kg": weight,
        },
        "vitals": vitals,
        "labs": labs,
        "self_report": self_report,
        "signal_history": history,
    }
    return {
        "patient_id": patient_id,
        "date": assessment_dt.date().isoformat(),
        "phase1_input": phase1_input,
        "patient_profile": {"weight_kg": weight, "age": age, "sex": overrides.get("sex", demo.get("sex", "unknown"))},
        "meal_intake": {
            "period": "previous_24h",
            "date": (assessment_dt.date() - timedelta(days=1)).isoformat(),
            "calories_kcal": float(nutrition.get("calories_kcal", 0)),
            "carbohydrates_g": float(nutrition.get("carbohydrates_g", 0)),
            "protein_g": float(nutrition.get("protein_g", 0)),
            "total_fat_g": float(nutrition.get("total_fat_g", 0)),
            "calcium_mg": float(nutrition.get("calcium_mg", 0)),
            "cholesterol_mg": float(nutrition.get("cholesterol_mg", 0)),
            "iron_mg": float(nutrition.get("iron_mg", 0)),
            "potassium_mg": float(nutrition.get("potassium_mg", 0)),
            "saturated_fat_g": float(nutrition.get("saturated_fat_g", 0)),
            "sodium_mg": float(nutrition.get("sodium_mg", 0)),
            "sugar_g": float(nutrition.get("sugar_g", 0)),
            "vitamin_c_mg": float(nutrition.get("vitamin_c_mg", 0)),
            "vitamin_d_ug": float(nutrition.get("vitamin_d_ug", 0)),
            "vitamin_e_mg": float(nutrition.get("vitamin_e_mg", 0)),
            "total_nutrient_value": float(nutrition.get("total_nutrient_value", 0)),
            "hydration_ml": float(nutrition.get("hydration_ml", 0)),
        },
    }

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: HMM-STYLE PHASE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

EMISSION_TABLES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "HR_trend": {"normal": (0.35, 0.25, 0.30, 0.45), "elevated": (0.20, 0.50, 0.25, 0.05), "sustained_tachycardia": (0.10, 0.70, 0.17, 0.03), "unknown": (1, 1, 1, 1)},
    "Temp_pattern": {"afebrile": (0.35, 0.20, 0.35, 0.50), "low_grade": (0.20, 0.55, 0.20, 0.05), "sustained_fever": (0.05, 0.75, 0.17, 0.03), "unknown": (1, 1, 1, 1)},
    "SpO2_level": {"normal": (0.35, 0.30, 0.35, 0.45), "borderline": (0.15, 0.35, 0.40, 0.10), "low": (0.05, 0.55, 0.30, 0.10), "unknown": (1, 1, 1, 1)},
    "CRP_velocity": {"rising": (0.15, 0.70, 0.12, 0.03), "plateau": (0.10, 0.50, 0.30, 0.10), "declining": (0.05, 0.15, 0.65, 0.15), "normal_low": (0.05, 0.08, 0.22, 0.65), "elevated_unknown_direction": (0.10, 0.55, 0.28, 0.07), "unknown": (1, 1, 1, 1)},
    "WBC_pattern": {"normal": (0.35, 0.25, 0.30, 0.45), "elevated": (0.15, 0.60, 0.20, 0.05), "rising": (0.10, 0.70, 0.17, 0.03), "declining": (0.05, 0.20, 0.60, 0.15), "unknown": (1, 1, 1, 1)},
    "Albumin_trend": {"normal_stable": (0.35, 0.25, 0.35, 0.45), "low": (0.10, 0.35, 0.45, 0.10), "declining": (0.10, 0.35, 0.40, 0.15), "unknown": (1, 1, 1, 1)},
    "Prealbumin_trend": {"rising": (0.03, 0.12, 0.70, 0.15), "flat_declining": (0.05, 0.35, 0.50, 0.10), "low": (0.10, 0.40, 0.40, 0.10), "normal_stable": (0.15, 0.20, 0.30, 0.35), "unknown": (1, 1, 1, 1)},
    "Glucose_level": {"controlled": (0.30, 0.30, 0.35, 0.45), "elevated": (0.10, 0.45, 0.35, 0.10), "severely_elevated": (0.05, 0.60, 0.30, 0.05), "unknown": (1, 1, 1, 1)},
    "Pain_trajectory": {"improving": (0.05, 0.20, 0.60, 0.15), "stable": (0.20, 0.40, 0.30, 0.10), "high_current_pain_unknown_trend": (0.10, 0.60, 0.25, 0.05), "low_current_pain_unknown_trend": (0.15, 0.15, 0.25, 0.45), "unknown": (1, 1, 1, 1)},
}


def day_prior(day: Optional[int]) -> List[float]:
    if day is None or day < 0:
        return [0.25, 0.25, 0.25, 0.25]
    if day == 0:
        return _normalize([0.85, 0.15, 0.0, 0.0])
    if day < 2:
        return _normalize([0.05, 0.88, 0.07, 0.0])
    if day < 5:
        return _normalize([0.01, 0.82, 0.16, 0.01])
    if day < 10:
        return _normalize([0.0, 0.30, 0.65, 0.05])
    if day < 21:
        return _normalize([0.0, 0.10, 0.78, 0.12])
    return _normalize([0.0, 0.04, 0.18, 0.78])


def estimate_healing_phase(state: Mapping[str, Any]) -> Dict[str, Any]:
    phase1 = state.get("phase1_input", state)
    day = int(phase1.get("day_post_op", 0))
    observed = phase1.get("observed_signals", {}) or {}
    prior = day_prior(day)
    likelihood = [1.0, 1.0, 1.0, 1.0]
    for name, category in observed.items():
        table = EMISSION_TABLES.get(name)
        if not table:
            continue
        probs = table.get(str(category), table.get("unknown", (1, 1, 1, 1)))
        for i in range(4):
            likelihood[i] *= max(probs[i], 1e-9)
    # Temperature dampens overconfidence while preserving the phase ranking.
    likelihood = [p ** 0.65 for p in likelihood]
    posterior = _normalize([prior[i] * likelihood[i] for i in range(4)])
    idx = max(range(4), key=lambda i: posterior[i])
    phase = PHASES[idx]
    flags = []
    expected_by_day = "haemostasis" if day == 0 else ("inflammation" if day < 5 else ("proliferation" if day < 21 else "remodelling"))
    if expected_by_day != phase:
        flags.append(f"DAY_SIGNAL_DISCORDANCE: day suggests {expected_by_day}, signal posterior suggests {phase}")
    if day >= 5 and phase == "inflammation":
        flags.append("PERSISTENT_INFLAMMATORY_SIGNAL_AFTER_DAY_5")
    return {
        "status": "completed",
        "model_version": "standalone_hmm_phase_estimator_v013",
        "current_phase": phase,
        "phase_confidence": round(posterior[idx], 4),
        "phase_posteriors": {PHASES[i]: round(posterior[i], 4) for i in range(4)},
        "hmm_features_used": observed,
        "day_prior": {PHASES[i]: round(prior[i], 4) for i in range(4)},
        "emission_likelihood": {PHASES[i]: round(likelihood[i], 4) for i in range(4)},
        "supporting_evidence": [f"{k}={v}" for k, v in observed.items()],
        "uncertainty_flags": flags,
        "limitations": [
            "No wound imaging or clinical wound assessment score supplied; phase is inferred from indirect signal trajectory.",
            "Demo probabilities are clinically initialised and should be calibrated with local outcome data before clinical deployment.",
        ],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: DEVIATION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_flag(deviation_type: str, signal_pattern: str, clinical_interpretation: str, alert_priority: str, evidence: List[str], escalation: List[str]) -> Dict[str, Any]:
    return {
        "deviation_type": deviation_type,
        "signal_pattern": signal_pattern,
        "clinical_interpretation": clinical_interpretation,
        "alert_priority": alert_priority,
        "evidence": evidence,
        "recommended_escalation": escalation,
    }


def expected_signal_band(signal: str, day: int, phase: str) -> Tuple[float, float, float]:
    # Demo cohort curves: expected value, lower bound, upper bound.
    if signal == "CRP":
        expected = 100 * math.exp(-0.18 * max(day - 2, 0)) if day >= 2 else 25 + day * 35
        return expected, max(0, expected - 45), expected + 45
    if signal == "WBC":
        expected = 11500 if phase == "inflammation" else (9000 if phase == "proliferation" else 7600)
        return expected, expected - 2800, expected + 2800
    if signal == "albumin":
        expected = 3.2 if phase == "proliferation" else (3.7 if phase == "remodelling" else 3.5)
        return expected, expected - 0.7, expected + 0.7
    if signal == "prealbumin":
        expected = 120 + min(max(day - 5, 0), 20) * 3 if phase in {"proliferation", "remodelling"} else 140
        return expected, expected - 55, expected + 55
    if signal == "glucose":
        return 130, 80, 180
    if signal == "spo2":
        return 97, 93, 100
    if signal == "heart_rate":
        return 86, 60, 110
    if signal == "temperature":
        return 37.1, 36.0, 38.5
    return 0, 0, 0


def run_deviation_detector(state: Mapping[str, Any], layer1: Mapping[str, Any]) -> Dict[str, Any]:
    """Responsive standalone Layer-2 deviation detector.

    This remains a demo approximation of the GP + clinical-rule layer. The signal
    table updates immediately when sidebar values change, and clinically important
    threshold breaches create flags even when a long trend history is unavailable.
    """
    phase1 = state["phase1_input"]
    day = int(phase1.get("day_post_op", 0))
    phase = str(layer1.get("current_phase", "")).lower()
    history = phase1.get("signal_history", {}) or {}
    labs = phase1.get("labs", {}) or {}
    vitals = phase1.get("vitals", {}) or {}
    profile = phase1.get("clinical_profile", {}) or {}

    # A light risk multiplier widens expected bands for higher-risk profiles,
    # mimicking the clinical-profile conditioning in the full GP layer.
    risk = 1.0
    if profile.get("diabetes_status"):
        risk += 0.10
    try:
        if float(profile.get("asa_score", 1)) >= 3:
            risk += 0.10
        if float(profile.get("bmi", 25)) >= 30:
            risk += 0.05
        if float(profile.get("nrs_2002_score", 0)) >= 3:
            risk += 0.05
    except Exception:
        pass

    predictions: Dict[str, Any] = {}
    for signal, rows in history.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: float(r.get("day_post_op", 0)))
        current = float(rows_sorted[-1]["value"])
        expected, lo, hi = expected_signal_band(signal, day, phase)
        # widen the interval around expected for higher-risk patients
        half_width = max(abs(expected - lo), abs(hi - expected)) * risk
        lo_risk = expected - half_width
        hi_risk = expected + half_width
        if signal in {"CRP", "WBC"}:
            lo_risk = max(0.0, lo_risk)
        if signal == "spo2":
            hi_risk = min(100.0, hi_risk)
        predictions[signal] = {
            "signal": signal,
            "observed_value": round(current, 4),
            "predicted_value": round(expected, 4),
            "credible_interval_95": {"lower": round(lo_risk, 4), "upper": round(hi_risk, 4)},
            "outside_credible_interval": current < lo_risk or current > hi_risk,
            "n_previous_observations": max(0, len(rows_sorted) - 1),
            "rule_sensitivity": "updates directly from current sidebar value plus simple trend history",
        }

    flags: List[Dict[str, Any]] = []
    single_day_reports: List[Dict[str, Any]] = []

    # v011: keep the clinical deviation gates strict, but still report
    # single-day outside-band observations separately for demo visibility.
    # These reports do not change Layer-2 priority or escalation unless one of
    # the clinically gated deviation rules below also fires.
    def add_single_day_report_if_outside(signal: str, clinical_gate: str, note: str) -> None:
        pred = predictions.get(signal) or {}
        if not pred.get("outside_credible_interval"):
            return
        ci = pred.get("credible_interval_95", {}) or {}
        observed = pred.get("observed_value")
        lower = ci.get("lower")
        upper = ci.get("upper")
        direction = "below" if lower is not None and observed is not None and observed < lower else "above"
        single_day_reports.append({
            "signal": signal,
            "observed_value": observed,
            "expected_value": pred.get("predicted_value"),
            "credible_interval_95": ci,
            "direction": direction,
            "status": "reported_single_day_observation",
            "clinical_gate_for_deviation_flag": clinical_gate,
            "note": note,
        })

    add_single_day_report_if_outside(
        "CRP",
        "Specific CRP deviation flag requires day_post_op > 3 plus CRP plateau/re-elevation or above-band trajectory.",
        "Report as a single-day inflammatory trajectory observation; do not escalate from this alone before the clinical gate is met.",
    )
    add_single_day_report_if_outside(
        "albumin",
        "Specific albumin context flag is used during proliferation/remodelling phases.",
        "Report as a single-day slow-marker observation; interpret with inflammation and nutrition context.",
    )
    add_single_day_report_if_outside(
        "prealbumin",
        "Specific prealbumin/protein-synthesis flag requires day_post_op >= 5.",
        "Report as a single-day protein-status observation; it becomes a stronger deviation flag after the clinical gate is met.",
    )
    add_single_day_report_if_outside(
        "WBC",
        "Specific WBC review flag requires post-Day-2 elevation threshold in this demo.",
        "Report as single-day immune activity outside expected band.",
    )
    add_single_day_report_if_outside(
        "spo2",
        "Specific oxygenation flag requires SpO₂ < 93 or borderline SpO₂ during proliferation.",
        "Report as single-day oxygenation observation.",
    )
    add_single_day_report_if_outside(
        "heart_rate",
        "Specific tachycardia flag requires HR > 100 after Day 2.",
        "Report as single-day heart-rate observation.",
    )
    add_single_day_report_if_outside(
        "temperature",
        "Specific fever flag requires temperature ≥ 38.0°C after Day 2 in this demo.",
        "Report as single-day temperature observation; sustained duration should be verified clinically.",
    )
    add_single_day_report_if_outside(
        "glucose",
        "Specific glucose flag requires glucose > 180 mg/dL.",
        "Report as single-day glucose observation.",
    )

    def latest(signal: str, default: Optional[float] = None) -> Optional[float]:
        vals = _latest_values(history, signal, 1)
        return vals[-1] if vals else default

    crp = latest("CRP", float(labs.get("CRP", 0) or 0))
    crp_vals = _latest_values(history, "CRP", 2)
    crp_days = _latest_times(history, "CRP", 2)
    crp_delta = crp_vals[-1] - crp_vals[-2] if len(crp_vals) >= 2 else None
    crp_pred = predictions.get("CRP", {})
    crp_outside = bool(crp_pred.get("outside_credible_interval")) and crp is not None and crp > crp_pred.get("credible_interval_95", {}).get("upper", 10**9)
    if day > 3 and crp is not None and (crp_outside or crp >= 80 or (crp_delta is not None and crp >= 10 and crp_delta >= -5)):
        evidence = [f"latest CRP={crp:.1f} mg/L"]
        if crp_delta is not None and len(crp_days) >= 2:
            evidence.extend([f"previous CRP={crp_vals[-2]:.1f} mg/L", f"delta={crp_delta:.1f} mg/L"])
        if crp_outside:
            evidence.append("CRP is above the expected trajectory band")
        flags.append(make_flag(
            "inflammation_not_resolving",
            "CRP plateau, re-elevation, or value above expected post-op trajectory after Day 3",
            "Inflammation may not be resolving as expected; wound/systemic complication should be reviewed.",
            "HIGH",
            evidence,
            ["same_day_physician_review"],
        ))

    inflammation_p = float((layer1.get("phase_posteriors") or {}).get("inflammation", 0.0))
    if day > 7 and (phase == "inflammation" or inflammation_p >= 0.45):
        flags.append(make_flag(
            "phase_transition_delay",
            "Phase posterior remains weighted toward inflammation beyond Day 7",
            "Possible delayed transition; review nutrition, glucose control, and infection risk.",
            "HIGH",
            [f"day_post_op={day}", f"current_phase={phase}", f"inflammation_posterior={inflammation_p:.4f}"],
            ["same_day_physician_review", "dietitian_review"],
        ))

    spo2 = latest("spo2", float(vitals.get("spo2", 97) or 97))
    if spo2 is not None:
        if spo2 < 93:
            flags.append(make_flag(
                "oxygenation_concern",
                "SpO₂ < 93%",
                "Systemic oxygen delivery may be insufficient for safe recovery monitoring; immediate clinical review is needed.",
                "HIGH",
                [f"latest SpO2={spo2:.1f}%"],
                ["respiratory_review", "same_day_physician_review"],
            ))
        elif phase == "proliferation" and 93 <= spo2 <= 95:
            flags.append(make_flag(
                "oxygenation_concern",
                "SpO₂ 93–95% during proliferation",
                "Suboptimal oxygen delivery to healing tissue may impair collagen synthesis.",
                "MODERATE",
                [f"latest SpO2={spo2:.1f}%"],
                ["respiratory_review", "position_optimisation"],
            ))

    pre_vals = _latest_values(history, "prealbumin", 2)
    pre_days = _latest_times(history, "prealbumin", 2)
    pre_latest = pre_vals[-1] if pre_vals else float(labs.get("prealbumin", 0) or 0)
    pre_delta = pre_vals[-1] - pre_vals[-2] if len(pre_vals) >= 2 else None
    if day >= 5 and (pre_latest < 150 or (pre_delta is not None and pre_delta <= 2)):
        evidence = [f"latest prealbumin={pre_latest:.1f} mg/L"]
        if pre_delta is not None and len(pre_days) >= 2:
            evidence.extend([f"previous prealbumin={pre_vals[-2]:.1f} mg/L", f"delta={pre_delta:.1f} mg/L"])
        flags.append(make_flag(
            "protein_synthesis_impairment",
            "Prealbumin low, flat, or declining from Day 5 onward",
            "Nutritional insufficiency during peak collagen synthesis is possible.",
            "HIGH",
            evidence,
            ["immediate_dietitian_review"],
        ))

    glucose = latest("glucose", float(labs.get("glucose", 120) or 120))
    glucose_vals = _latest_values(history, "glucose", 2)
    if glucose is not None and glucose > 180:
        priority = "HIGH" if glucose < 250 else "CRITICAL"
        evidence = [f"latest glucose={glucose:.1f} mg/dL"]
        if len(glucose_vals) >= 2:
            evidence.append("latest readings=" + ", ".join(f"{v:.1f}" for v in glucose_vals[-2:]) + " mg/dL")
        flags.append(make_flag(
            "glucose_dysregulation",
            "Glucose > 180 mg/dL",
            "Hyperglycaemia may impair neutrophil function and increase infection risk.",
            priority,
            evidence,
            ["endocrine_or_pharmacy_review", "same_day_physician_review"],
        ))

    hr = latest("heart_rate", float(vitals.get("heart_rate", 85) or 85))
    if day > 2 and hr is not None and hr > 100:
        priority = "HIGH" if hr > 110 else "MODERATE"
        flags.append(make_flag(
            "tachycardia_review",
            "Heart rate > 100 after Day 2",
            "Tachycardia may reflect pain, hypovolaemia, infection, or pulmonary complication; persistence should be checked.",
            priority,
            [f"latest HR={hr:.1f} bpm"],
            ["repeat_vitals", "physician_review_if_persistent"],
        ))

    temp = latest("temperature", float(vitals.get("temperature", 37.0) or 37.0))
    if day > 2 and temp is not None and temp >= 38.0:
        if temp >= 39.0:
            priority = "CRITICAL"
        elif temp >= 38.5:
            priority = "HIGH"
        else:
            priority = "MODERATE"
        flags.append(make_flag(
            "fever_pattern",
            "Temperature ≥ 38.0°C after Day 2",
            "Fever pattern may indicate surgical site infection or systemic complication; duration should be verified.",
            priority,
            [f"latest temperature={temp:.2f}°C"],
            ["repeat_temperature", "same_day_physician_review_if_persistent"],
        ))

    wbc = latest("WBC", float(labs.get("WBC", 9000) or 9000))
    if day > 2 and wbc is not None and wbc > 13000:
        flags.append(make_flag(
            "wbc_elevation_review",
            "WBC > 13,000 after Day 2",
            "Leukocytosis may reflect infection or persistent inflammatory activity; interpret with CRP, temperature, and wound assessment.",
            "MODERATE",
            [f"latest WBC={wbc:.0f} cells/µL"],
            ["physician_review_if_persistent"],
        ))

    albumin = latest("albumin", float(labs.get("albumin", 3.5) or 3.5))
    if phase in {"proliferation", "remodelling"} and albumin is not None and albumin < 3.2:
        flags.append(make_flag(
            "albumin_low_context",
            "Albumin < 3.2 g/dL during tissue repair phase",
            "Low albumin may indicate reduced protein reserve or inflammation burden; interpret as context, not an acute response marker.",
            "MODERATE",
            [f"latest albumin={albumin:.1f} g/dL"],
            ["dietitian_review_next_round"],
        ))

    zinc = float(labs.get("zinc", 0) or 0)
    if zinc and zinc < 60:
        flags.append(make_flag(
            "zinc_deficiency_context",
            "Serum zinc < 60",
            "Zinc deficiency can impair epithelialisation and immune function; nutrition review should include zinc context.",
            "MODERATE",
            [f"serum zinc={zinc:.1f}"],
            ["dietitian_review_next_round"],
        ))


    overall = max([f["alert_priority"] for f in flags] or ["NONE"], key=lambda p: SEVERITY_ORDER.get(p, 0))
    escalate = sorted({role for f in flags for role in f.get("recommended_escalation", [])})
    return {
        "status": "completed",
        "model_version": "standalone_deviation_detector_v013_clinical_rules_plus_single_day_reporting",
        "day_post_op": int(day),
        "current_phase_from_layer1": phase,
        "phase_posteriors_from_layer1": layer1.get("phase_posteriors", {}),
        "signal_predictions": predictions,
        "deviation_flags": flags,
        "single_day_reports": single_day_reports,
        "overall_alert_priority": overall,
        "escalation_required": overall in {"MODERATE", "HIGH", "CRITICAL"},
        "escalate_to": escalate,
        "rule_engine_note": "Clinical deviation flags follow gated rules; single-day outside-band observations are reported separately for demo visibility.",
        "limitations": [
            "Daily-summary demo cannot prove sustained fever/tachycardia duration unless duration fields are supplied.",
            "Trajectory bands are demo-initialised and should be calibrated with local outcome data before deployment.",
            "No wound imaging or clinical wound assessment score is used here; wound-bed status cannot be directly observed.",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: NUTRITIONAL GAP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

TARGETS = {
    "inflammation": {
        "protein": ("Protein", 1.5, 2.0, "g/kg/day", 0.30),
        "calories": ("Total calories", 25.0, 30.0, "kcal/kg/day", 0.20),
        "vitamin_c": ("Vitamin C", 200.0, 500.0, "mg/day", 0.15),
        "hydration": ("Hydration", 30.0, 35.0, "ml/kg/day", 0.10),
    },
    "proliferation": {
        "protein": ("Protein", 1.8, 2.0, "g/kg/day", 0.35),
        "calories": ("Total calories", 30.0, 35.0, "kcal/kg/day", 0.20),
        "vitamin_c": ("Vitamin C", 500.0, 1000.0, "mg/day", 0.15),
        "hydration": ("Hydration", 30.0, 35.0, "ml/kg/day", 0.10),
    },
    "remodelling": {
        "protein": ("Protein", 1.2, 1.5, "g/kg/day", 0.30),
        "calories": ("Total calories", 25.0, 30.0, "kcal/kg/day", 0.25),
        "vitamin_c": ("Vitamin C", 100.0, 200.0, "mg/day", 0.15),
        "hydration": ("Hydration", 30.0, 30.0, "ml/kg/day", 0.20),
    },
    "haemostasis": {
        "hydration": ("Hydration", 30.0, 35.0, "ml/kg/day", 1.00),
    },
}


def score_one(nutrient: str, label: str, target_min: float, target_max: float, unit: str, weight: float, actual_value: Optional[float], source_field: str, normalized: bool, weight_kg: Optional[float]) -> Dict[str, Any]:
    target = {"target_min": target_min, "target_max": target_max, "unit": unit, "scoring_weight": weight}
    actual = {"value": clean_float(actual_value, 4), "unit": unit, "source_field": source_field, "normalized_from_total": normalized, "body_weight_kg_used": clean_float(weight_kg, 4) if normalized else None}
    if actual_value is None:
        return {"nutrient": nutrient, "display_name": label, "target": target, "actual": actual, "gap_fraction": None, "gap_pct": None, "adequacy_ratio_capped": None, "priority": "LOW", "status": "missing_intake", "interpretation": f"{label} intake could not be scored.", "recommended_escalation": ["dietitian_intake_log_review"]}
    gap = (target_min - actual_value) / target_min
    adequacy = max(0.0, min(1.0, actual_value / target_min))
    if gap > 0.30:
        priority, status, escalation = "HIGH", "high_gap", ["immediate_dietitian_review"]
        interp = f"{label} intake is more than 30% below the phase target."
    elif gap > 0.15:
        priority, status, escalation = "MODERATE", "moderate_gap", ["dietitian_review_next_round"]
        interp = f"{label} intake is more than 15% below the phase target."
    elif gap > 0.0:
        priority, status, escalation = "LOW", "minor_gap", ["routine_dietitian_follow_up"]
        interp = f"{label} intake is slightly below the phase target."
    else:
        priority, status, escalation = "NONE", "adequate", []
        interp = f"{label} intake meets the lower phase target."
    return {"nutrient": nutrient, "display_name": label, "target": target, "actual": actual, "gap_fraction": round(gap, 4), "gap_pct": round(gap * 100, 1), "adequacy_ratio_capped": round(adequacy, 4), "priority": priority, "status": status, "interpretation": interp, "recommended_escalation": escalation}




def classify_contextual_nutrition_observations(intake: Mapping[str, Any], calories_kcal: float, carb_pct: Optional[float]) -> List[Dict[str, Any]]:
    """Create visible context notes from non-scored meal totals.

    These observations are deliberately separate from wound-gap scoring. They
    make the extra nutrition fields reactive in the UI without pretending that
    every micronutrient has a validated phase-specific wound-healing target.
    """
    notes: List[Dict[str, Any]] = []

    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(intake.get(key, default) or default)
        except Exception:
            return default

    protein_g = f("protein_g")
    carbs_g = f("carbohydrates_g")
    fat_g = f("total_fat_g")
    sugar_g = f("sugar_g")
    sodium_mg = f("sodium_mg")
    saturated_fat_g = f("saturated_fat_g")
    vitamin_d_ug = f("vitamin_d_ug")
    vitamin_e_mg = f("vitamin_e_mg")
    iron_mg = f("iron_mg")
    potassium_mg = f("potassium_mg")
    calcium_mg = f("calcium_mg")

    if calories_kcal <= 0:
        notes.append({"area": "overall_intake", "status": "no_energy_logged", "priority": "HIGH", "observation": "No calories were logged for the previous 24 hours.", "use": "Context for dietitian intake-log review."})
        return notes

    fat_pct = fat_g * 9.0 / calories_kcal * 100.0 if calories_kcal > 0 else None
    sugar_pct = sugar_g * 4.0 / calories_kcal * 100.0 if calories_kcal > 0 else None

    if carb_pct is not None and carb_pct > 55.0:
        notes.append({"area": "carbohydrate_distribution", "status": "high_carb_share", "priority": "MODERATE", "observation": f"Carbohydrates provide {carb_pct:.1f}% of calories.", "use": "Recipe selector may shift toward protein/fibre balance, especially if glucose is high."})
    if sugar_pct is not None and sugar_g > 50.0 and sugar_pct > 10.0:
        notes.append({"area": "sugar", "status": "high_sugar_context", "priority": "MODERATE", "observation": f"Sugar is {sugar_g:.0f} g, about {sugar_pct:.1f}% of calories.", "use": "Avoid sugar-heavy recovery drinks unless approved."})
    if fat_pct is not None and fat_pct < 15.0 and calories_kcal < 1800:
        notes.append({"area": "fat_energy_density", "status": "low_fat_low_energy_context", "priority": "LOW", "observation": f"Fat is {fat_g:.0f} g and total calories are low.", "use": "Energy-density review may be useful if tolerated and permitted."})
    if saturated_fat_g > 25.0:
        notes.append({"area": "saturated_fat", "status": "high_saturated_fat_context", "priority": "LOW", "observation": f"Saturated fat is {saturated_fat_g:.0f} g.", "use": "Context only; not a wound-healing gap flag."})
    if sodium_mg > 2300.0:
        notes.append({"area": "sodium", "status": "high_sodium_context", "priority": "LOW", "observation": f"Sodium is {sodium_mg:.0f} mg.", "use": "Context for fluid/BP-sensitive patients; physician/dietitian decides relevance."})
    if vitamin_d_ug == 0:
        notes.append({"area": "vitamin_d", "status": "not_logged", "priority": "LOW", "observation": "Vitamin D is not logged.", "use": "Tracked for completeness; not part of the phase-specific wound-gap score in this demo."})
    elif vitamin_d_ug < 10.0:
        notes.append({"area": "vitamin_d", "status": "low_context", "priority": "LOW", "observation": f"Vitamin D intake is {vitamin_d_ug:.1f} µg.", "use": "Context only unless local protocol defines a target."})
    if vitamin_e_mg == 0:
        notes.append({"area": "vitamin_e", "status": "not_logged", "priority": "LOW", "observation": "Vitamin E is not logged.", "use": "Tracked for completeness; not part of the phase-specific wound-gap score in this demo."})
    if iron_mg > 0 and iron_mg < 8.0:
        notes.append({"area": "iron", "status": "low_context", "priority": "LOW", "observation": f"Iron intake is {iron_mg:.1f} mg.", "use": "Context only; review if anaemia or local protocol applies."})
    if potassium_mg > 0 and potassium_mg < 2000.0:
        notes.append({"area": "potassium", "status": "low_context", "priority": "LOW", "observation": f"Potassium intake is {potassium_mg:.0f} mg.", "use": "Context only; renal/cardiac restrictions must be checked."})
    if calcium_mg > 0 and calcium_mg < 600.0:
        notes.append({"area": "calcium", "status": "low_context", "priority": "LOW", "observation": f"Calcium intake is {calcium_mg:.0f} mg.", "use": "Context only; not a wound-gap flag."})

    if not notes:
        notes.append({"area": "context", "status": "no_context_alerts", "priority": "NONE", "observation": "Additional tracked meal totals do not trigger contextual notes in this demo run.", "use": "Continue routine monitoring."})
    return notes


def find_dominant_nutritional_gap(scores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Choose the gap that should drive the clinical insight and recipe band."""
    clinical_order = {"protein": 4, "calories": 3, "hydration": 2, "vitamin_c": 1}
    candidate_scores = []
    for s in scores:
        gap = s.get("gap_pct")
        if gap is None:
            continue
        try:
            gap_f = float(gap)
        except Exception:
            continue
        priority = str(s.get("priority", "NONE"))
        if priority == "NONE" or gap_f <= 0:
            continue
        nutrient = str(s.get("nutrient", ""))
        candidate_scores.append((SEVERITY_ORDER.get(priority, 0), gap_f, clinical_order.get(nutrient, 0), s))
    if not candidate_scores:
        return {"nutrient": None, "display_name": "No major nutrition gap", "gap_pct": 0.0, "priority": "NONE", "status": "adequate"}
    candidate_scores.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    chosen = dict(candidate_scores[0][3])
    return {"nutrient": chosen.get("nutrient"), "display_name": chosen.get("display_name"), "gap_pct": chosen.get("gap_pct"), "priority": chosen.get("priority"), "status": chosen.get("status")}

def run_nutritional_gap_engine(state: Mapping[str, Any], layer1: Mapping[str, Any], layer2: Mapping[str, Any]) -> Dict[str, Any]:
    phase = str(layer1.get("current_phase", "proliferation")).lower()
    targets = TARGETS.get(phase, TARGETS["proliferation"])
    intake = state.get("meal_intake", {}) or {}
    profile = state.get("patient_profile", {}) or {}
    phase1 = state.get("phase1_input", {}) or {}
    clinical_profile = phase1.get("clinical_profile", {}) or {}
    weight_kg = float(profile.get("weight_kg") or clinical_profile.get("body_weight_kg") or 0) or None

    protein = (float(intake.get("protein_g", 0)) / weight_kg) if weight_kg else None
    calories = (float(intake.get("calories_kcal", 0)) / weight_kg) if weight_kg else None
    hydration = (float(intake.get("hydration_ml", 0)) / weight_kg) if weight_kg else None
    vitc = float(intake.get("vitamin_c_mg", 0)) if "vitamin_c_mg" in intake else None

    actuals = {
        "protein": (protein, "protein_g", True),
        "calories": (calories, "calories_kcal", True),
        "vitamin_c": (vitc, "vitamin_c_mg", False),
        "hydration": (hydration, "hydration_ml", True),
    }
    scores = []
    for nutrient, values in targets.items():
        label, tmin, tmax, unit, wt = values
        av, src, norm = actuals.get(nutrient, (None, nutrient, False))
        scores.append(score_one(nutrient, label, tmin, tmax, unit, wt, av, src, norm, weight_kg))

    weighted, wsum = 0.0, 0.0
    for s in scores:
        adequacy = s.get("adequacy_ratio_capped")
        if adequacy is not None:
            wt = float(s["target"].get("scoring_weight", 0))
            weighted += wt * float(adequacy)
            wsum += wt
    nss = weighted / wsum if wsum else None
    if nss is None:
        nss_status, nss_escalation = "not_computable", []
    elif nss < 0.70:
        nss_status, nss_escalation = "inadequate_immediate_review", ["immediate_dietitian_review"]
    elif nss < 0.85:
        nss_status, nss_escalation = "borderline_review_next_round", ["dietitian_review_next_round"]
    else:
        nss_status, nss_escalation = "adequate", []

    calories_kcal = float(intake.get("calories_kcal", 0) or 0)
    carbs_g = float(intake.get("carbohydrates_g", 0) or 0)
    carb_pct = (carbs_g * 4.0 / calories_kcal * 100.0) if calories_kcal > 0 else None
    high_carb = carb_pct is not None and carb_pct > 55.0

    by = {s["nutrient"]: s for s in scores}
    flags = [str(f.get("deviation_type", "")) for f in layer2.get("deviation_flags", [])]
    interactions: List[Dict[str, Any]] = []
    if by.get("protein", {}).get("gap_fraction") is not None and by["protein"]["gap_fraction"] > 0.30 and "inflammation_not_resolving" in flags:
        interactions.append({"interaction_type": "crp_not_resolving_plus_protein_gap", "alert_priority": "HIGH", "combined_interpretation": "Insufficient substrate for immune function may contribute to inflammation persistence.", "alert_text": f"Protein intake {by['protein']['gap_pct']}% below phase target. CRP not resolving. Dietitian/physician review required.", "recommended_escalation": ["immediate_dietitian_review", "same_day_physician_review"]})
    if by.get("vitamin_c", {}).get("gap_fraction") is not None and by["vitamin_c"]["gap_fraction"] > 0 and "phase_transition_delay" in flags:
        interactions.append({"interaction_type": "phase_transition_delay_plus_vitamin_c_gap", "alert_priority": "HIGH", "combined_interpretation": "Collagen synthesis substrate may be insufficient.", "alert_text": "Vitamin C intake below phase threshold with delayed phase transition. Review required.", "recommended_escalation": ["dietitian_review", "same_day_physician_review"]})
    if high_carb and "glucose_dysregulation" in flags:
        interactions.append({"interaction_type": "glucose_dysregulation_plus_high_carbohydrate_load", "alert_priority": "HIGH", "combined_interpretation": "Carbohydrate load may be contributing to glycaemic instability.", "alert_text": "Glucose dysregulation with high carbohydrate load. Dietitian/pharmacy review required.", "recommended_escalation": ["endocrine_or_pharmacy_review", "dietitian_review"]})
    if by.get("calories", {}).get("gap_fraction") is not None and by["calories"]["gap_fraction"] > 0.20 and "protein_synthesis_impairment" in flags:
        interactions.append({"interaction_type": "prealbumin_flat_plus_caloric_gap", "alert_priority": "HIGH", "combined_interpretation": "Insufficient energy intake may cause protein to be used for energy instead of tissue synthesis.", "alert_text": f"Caloric intake {by['calories']['gap_pct']}% below target with prealbumin concern. Dietitian review required.", "recommended_escalation": ["immediate_dietitian_review"]})

    priorities = [s["priority"] for s in scores] + [i["alert_priority"] for i in interactions]
    if nss is not None and nss < 0.70:
        priorities.append("HIGH")
    elif nss is not None and nss < 0.85:
        priorities.append("MODERATE")
    overall = max(priorities or ["NONE"], key=lambda p: SEVERITY_ORDER.get(p, 0))
    escalate = sorted({role for s in scores for role in s.get("recommended_escalation", [])} | {role for i in interactions for role in i.get("recommended_escalation", [])} | set(nss_escalation))
    dominant_gap = find_dominant_nutritional_gap(scores)
    contextual_observations = classify_contextual_nutrition_observations(intake, calories_kcal, carb_pct)

    return {
        "status": "completed",
        "model_version": "standalone_rule_based_nutritional_gap_engine_v014",
        "current_phase_from_layer1": phase,
        "previous_24h_intake": {**intake, "protein_g_per_kg": clean_float(protein, 3), "calories_kcal_per_kg": clean_float(calories, 3), "hydration_ml_per_kg": clean_float(hydration, 3), "carbohydrate_percent_calories": clean_float(carb_pct, 1)},
        "active_phase_targets": {k: {"display_name": v[0], "target_min": v[1], "target_max": v[2], "unit": v[3], "scoring_weight": v[4]} for k, v in targets.items()},
        "scored_nutritional_gaps": scores,
        "nutritional_gap_flags": [s for s in scores if s["priority"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}],
        "dominant_nutritional_gap": dominant_gap,
        "contextual_nutrition_observations": contextual_observations,
        "nutritional_sufficiency_score": round(nss, 4) if nss is not None else None,
        "nutritional_sufficiency_status": nss_status,
        "interaction_flags": interactions,
        "overall_alert_priority": overall,
        "escalation_required": overall in {"MODERATE", "HIGH", "CRITICAL"} or (nss is not None and nss < 0.85),
        "escalate_to": escalate,
        "limitations": ["Nutritional outputs are review flags only and are not autonomous diet prescriptions."],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE + RECIPES
# ═══════════════════════════════════════════════════════════════════════════════

def run_three_layer_pipeline(state: Dict[str, Any]) -> Dict[str, Any]:
    working = copy.deepcopy(state)
    l1 = estimate_healing_phase(working)
    working["phase1_input"]["phase1_healing_phase_layer"] = l1
    l2 = run_deviation_detector(working, l1)
    working["phase1_input"]["phase1_deviation_detector_layer"] = l2
    l3 = run_nutritional_gap_engine(working, l1, l2)
    working["phase1_nutritional_gap_engine_layer"] = l3
    return {"state_after_layer3": working, "layer1": l1, "layer2": l2, "layer3": l3}


def build_clinical_narrative(result: Mapping[str, Any]) -> str:
    l1, l2, l3 = result["layer1"], result["layer2"], result["layer3"]
    phase = l1.get("current_phase")
    confidence = l1.get("phase_confidence")
    priority = l3.get("overall_alert_priority") or l2.get("overall_alert_priority")
    high_gaps = [g for g in l3.get("scored_nutritional_gaps", []) if g.get("priority") == "HIGH"]
    mod_gaps = [g for g in l3.get("scored_nutritional_gaps", []) if g.get("priority") == "MODERATE"]
    interactions = l3.get("interaction_flags", [])
    lines = [
        f"Current estimated wound-healing phase is {phase} with confidence {confidence}.",
        f"Overall monitoring priority is {priority}. This is a review flag, not an autonomous treatment instruction.",
    ]
    if high_gaps:
        lines.append("High nutritional gaps detected: " + ", ".join(f"{g['display_name']} ({g.get('gap_pct')}% gap)" for g in high_gaps) + ".")
    if mod_gaps:
        lines.append("Moderate nutritional gaps detected: " + ", ".join(f"{g['display_name']} ({g.get('gap_pct')}% gap)" for g in mod_gaps) + ".")
    if interactions:
        lines.append("Combined trajectory + nutrition flags: " + "; ".join(i.get("alert_text", i.get("interaction_type", "")) for i in interactions) + ".")
    lines.append("Note: No wound imaging is available in this demo; phase is inferred from indirect signal trajectory.")
    return "\n\n".join(lines)


def build_macro_recipe_recommendation(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Choose patient-facing demo recipes from the current macro pattern.

    v012 fixes the previous behaviour where a high carbohydrate percentage could
    hide a severe low-protein or low-calorie state. The selection now follows the
    clinically dominant gap first, and uses carbohydrate distribution only when
    glucose/carb risk is genuinely the main issue.
    """
    l1 = result["layer1"]
    l2 = result["layer2"]
    l3 = result["layer3"]
    phase = str(l1.get("current_phase", "unknown")).lower()
    intake = l3.get("previous_24h_intake", {}) or {}
    scores = {g.get("nutrient"): g for g in l3.get("scored_nutritional_gaps", []) or []}
    interactions = l3.get("interaction_flags", []) or []
    dominant = l3.get("dominant_nutritional_gap", {}) or {}

    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    protein = as_float(intake.get("protein_g_per_kg"))
    calories = as_float(intake.get("calories_kcal_per_kg"))
    hydration = as_float(intake.get("hydration_ml_per_kg"))
    carb_pct_raw = intake.get("carbohydrate_percent_calories")
    carb_pct = as_float(carb_pct_raw, -1.0) if carb_pct_raw is not None else None
    sugar_g = as_float(intake.get("sugar_g"), 0.0)

    protein_gap = as_float(scores.get("protein", {}).get("gap_pct"), 0.0)
    calorie_gap = as_float(scores.get("calories", {}).get("gap_pct"), 0.0)
    vitc_gap = as_float(scores.get("vitamin_c", {}).get("gap_pct"), 0.0)
    hydration_gap = as_float(scores.get("hydration", {}).get("gap_pct"), 0.0)
    dominant_nutrient = str(dominant.get("nutrient") or "").lower()
    dominant_gap = as_float(dominant.get("gap_pct"), 0.0)

    glucose_or_carb_flag = any(
        i.get("interaction_type") == "glucose_dysregulation_plus_high_carbohydrate_load"
        for i in interactions
    ) or any(
        str(f.get("deviation_type", "")).lower() == "glucose_dysregulation"
        for f in l2.get("deviation_flags", []) or []
    )

    # Important: a low total intake can make carbohydrate percentage look high.
    # In that situation, the correct demo behaviour is to fix protein/energy first,
    # not to show only carbohydrate-review recipes.
    severe_low_overall = calorie_gap > 30.0 and protein_gap > 30.0
    high_carb_is_primary = glucose_or_carb_flag or ((carb_pct is not None and carb_pct > 55.0) and not severe_low_overall and calorie_gap <= 30.0) or sugar_g > 60.0

    if severe_low_overall:
        band = "Low overall intake support"
        status = "protein_energy_hydration_gap_pattern"
        trigger = f"Protein gap {protein_gap:.1f}% and calorie gap {calorie_gap:.1f}% are both high; this overrides carbohydrate-percentage selection."
        clinical_insight = "Protein and energy intake are both substantially below the current wound-healing phase target. Physician/dietitian review should prioritise whether intake can be safely advanced before focusing on carbohydrate distribution."
        recipes = [
            {
                "title": "Protein-energy recovery khichdi",
                "meal_slot": "Lunch or dinner",
                "patient_text": "Soft moong dal khichdi with paneer or tofu, curd on the side, and a small amount of ghee or olive oil if the care team allows.",
                "review_reason": "Addresses the combined protein and energy gap in one gentle meal pattern.",
                "ingredients": ["moong dal", "rice", "paneer or tofu", "curd", "ghee or olive oil if allowed", "soft vegetables"],
            },
            {
                "title": "Small high-protein add-on",
                "meal_slot": "Between meals",
                "patient_text": "Curd with roasted chana powder, or unsweetened lassi with soft paneer/tofu bites if tolerated.",
                "review_reason": "Adds protein without relying on a large meal when appetite is low.",
                "ingredients": ["curd", "roasted chana", "paneer or tofu", "water", "jeera"],
            },
        ]
    elif dominant_nutrient == "protein" or protein_gap > 15.0:
        band = "Low protein support"
        status = "protein_gap_pattern"
        trigger = f"Protein is {protein_gap:.1f}% below the current phase target."
        clinical_insight = "Protein is the dominant gap. A soft, protein-dense option is preferable for review."
        recipes = [
            {
                "title": "Moong dal paneer/tofu khichdi",
                "meal_slot": "Lunch or dinner",
                "patient_text": "Soft moong dal khichdi with paneer or tofu cubes, plus curd if tolerated.",
                "review_reason": "Improves protein density in a recovery-friendly texture.",
                "ingredients": ["moong dal", "rice", "paneer or tofu", "curd", "soft vegetables"],
            },
            {
                "title": "Curd + roasted chana snack",
                "meal_slot": "Evening snack",
                "patient_text": "Curd with roasted chana powder or soft chana, if chewing and digestion are comfortable.",
                "review_reason": "Adds protein between meals without a heavy main meal.",
                "ingredients": ["curd", "roasted chana", "jeera", "small pinch of salt if allowed"],
            },
        ]
    elif dominant_nutrient == "calories" or calorie_gap > 15.0:
        band = "Low calorie / energy support"
        status = "calorie_gap_pattern"
        trigger = f"Calories are {calorie_gap:.1f}% below the current phase target."
        clinical_insight = "Energy is the dominant gap. Adequate calories help preserve protein for tissue repair rather than energy use."
        recipes = [
            {
                "title": "Energy-support dalia bowl",
                "meal_slot": "Breakfast or dinner",
                "patient_text": "Soft dalia cooked with dal, vegetables, and a small amount of ghee or olive oil if allowed.",
                "review_reason": "Increases energy density while keeping the meal gentle.",
                "ingredients": ["dalia", "dal", "carrot or pumpkin", "ghee or olive oil if allowed"],
            },
            {
                "title": "Banana-curd recovery smoothie",
                "meal_slot": "Mid-morning",
                "patient_text": "Curd blended with banana; use unsweetened curd and avoid added sugar.",
                "review_reason": "Adds calories and protein in an easy-to-consume form when appetite is low.",
                "ingredients": ["curd", "banana", "milk or water", "cinnamon if allowed"],
            },
        ]
    elif high_carb_is_primary:
        band = "High carbohydrate / glucose review"
        status = "high_carbohydrate_pattern"
        trigger = "Carbohydrate/sugar pattern is high or glucose dysregulation is present."
        clinical_insight = "Review carbohydrate distribution while preserving adequate recovery calories. Do not create an automatic calorie deficit during active wound healing."
        recipes = [
            {
                "title": "Protein-balanced soft thali",
                "meal_slot": "Lunch or dinner",
                "patient_text": "Choose a smaller rice or roti portion, add extra dal or paneer/tofu, cooked vegetables, and curd if allowed.",
                "review_reason": "Shifts the meal toward protein and fibre while keeping energy available for recovery.",
                "ingredients": ["rice or roti", "dal", "paneer or tofu", "cooked vegetables", "curd"],
            },
            {
                "title": "Vitamin-C snack without sweet drinks",
                "meal_slot": "Snack",
                "patient_text": "Amla or guava slices, or cucumber/bell pepper salad with lemon. Avoid sweet juice unless the care team approves.",
                "review_reason": "Supports Vitamin C intake without increasing sugar-heavy beverages.",
                "ingredients": ["amla or guava", "cucumber", "bell pepper", "lemon", "jeera"],
            },
        ]
    elif dominant_nutrient == "hydration" or hydration_gap > 0.0:
        band = "Low hydration support"
        status = "hydration_gap_pattern"
        trigger = f"Hydration is {hydration_gap:.1f}% below the current phase target."
        clinical_insight = "Fluid intake is below the target; tolerance and restrictions should be reviewed."
        recipes = [
            {
                "title": "Clear soup schedule",
                "meal_slot": "Between meals",
                "patient_text": "Small portions of clear vegetable soup across the day if fluids are allowed.",
                "review_reason": "Improves fluid intake without forcing large volumes at once.",
                "ingredients": ["clear vegetable stock", "bottle gourd or carrot", "small pinch of salt if allowed"],
            },
            {
                "title": "Buttermilk or curd drink",
                "meal_slot": "Afternoon",
                "patient_text": "Unsweetened buttermilk or thin curd drink if tolerated and permitted.",
                "review_reason": "Supports hydration and tolerance, especially when appetite is low.",
                "ingredients": ["curd", "water", "roasted cumin", "mint if allowed"],
            },
        ]
    elif dominant_nutrient == "vitamin_c" or vitc_gap > 15.0:
        band = "Low Vitamin C support"
        status = "vitamin_c_gap_pattern"
        trigger = f"Vitamin C is {vitc_gap:.1f}% below the current phase target."
        clinical_insight = "Vitamin C is the dominant gap and matters for collagen-support review, especially during proliferation."
        recipes = [
            {
                "title": "Amla-guava Vitamin C plate",
                "meal_slot": "Mid-morning",
                "patient_text": "Amla or guava slices with a small pinch of salt/jeera if allowed.",
                "review_reason": "Adds Vitamin C substrate without needing a large meal.",
                "ingredients": ["amla", "guava", "jeera", "small pinch of salt if allowed"],
            },
            {
                "title": "Lemon vegetable salad",
                "meal_slot": "With lunch",
                "patient_text": "Cucumber, bell pepper, and lemon dressing; keep it soft/cooked if raw foods are restricted.",
                "review_reason": "Adds Vitamin C alongside the main meal.",
                "ingredients": ["cucumber", "bell pepper", "lemon", "soft cooked vegetables if raw foods are restricted"],
            },
        ]
    else:
        band = "Balanced macro support"
        status = "balanced_pattern"
        trigger = "Protein, calories, Vitamin C and hydration are close to the current phase targets."
        clinical_insight = "Current intake is close to phase targets; maintain balanced recovery meals and continue monitoring."
        recipes = [
            {
                "title": "Balanced recovery thali",
                "meal_slot": "Lunch or dinner",
                "patient_text": "Dal or lean protein, rice or roti, cooked vegetables, curd, and one Vitamin-C-rich fruit portion.",
                "review_reason": "Maintains protein, energy, micronutrients, and hydration without unnecessary escalation.",
                "ingredients": ["dal or lean protein", "rice or roti", "cooked vegetables", "curd", "guava or orange"],
            },
            {
                "title": "Protein + fruit snack",
                "meal_slot": "Snack",
                "patient_text": "Curd or paneer/tofu bites with guava, orange, or amla if allowed.",
                "review_reason": "Maintains protein and Vitamin C coverage between meals.",
                "ingredients": ["curd", "paneer or tofu", "guava, orange, or amla"],
            },
        ]

    if carb_pct is None or carb_pct < 0:
        macro_line = f"Protein: {protein:.2f} g/kg/day; calories: {calories:.1f} kcal/kg/day; hydration: {hydration:.1f} ml/kg/day."
    else:
        macro_line = f"Protein: {protein:.2f} g/kg/day; calories: {calories:.1f} kcal/kg/day; carbohydrate share: {carb_pct:.1f}% of calories; hydration: {hydration:.1f} ml/kg/day."

    return {
        "phase": phase,
        "band": band,
        "status": status,
        "trigger": trigger,
        "macro_line": macro_line,
        "dominant_gap": dominant,
        "clinical_insight": clinical_insight,
        "recipes": recipes,
        "safety_note": "Patient-facing demo wording only. Physician/dietitian approval is required before changing any diet plan.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("#  Phase 1 Wound/Tissue Recovery Demo")
st.caption("Standalone inpatient wound monitoring demo: Layer 1 HMM-style phase estimation → Layer 2 responsive trajectory deviation detection → Layer 3 rule-based nutrition gap scoring.")

with st.sidebar:
    st.header("Demo Controls")
    scenario_key = st.selectbox("Walkthrough", list(SCENARIOS.keys()), format_func=lambda k: SCENARIOS[k]["label"], index=0)
    demo = SCENARIOS[scenario_key]
    st.info(demo["description"])

    st.divider()
    st.subheader("Patient + surgery")
    patient_id = st.text_input("Patient ID", demo["patient_id"])
    age = st.number_input("Age", 18, 95, int(demo["age"]))
    sex = st.selectbox("Sex", ["unknown", "female", "male"], index=["unknown", "female", "male"].index(demo.get("sex", "unknown")) if demo.get("sex", "unknown") in ["unknown", "female", "male"] else 0)
    surgery_type = st.selectbox("Surgery type", ["abdominal", "orthopedic", "cardiac", "general"], index=["abdominal", "orthopedic", "cardiac", "general"].index(demo["surgery_type"]))
    day_post_op = st.slider("Post-op day (integer)", 0, 30, int(demo["day_post_op"]), step=1)
    weight_kg = st.number_input("Weight (kg)", 35.0, 180.0, float(demo["weight_kg"]), step=1.0)
    bmi = st.number_input("BMI", 14.0, 55.0, float(demo["bmi"]), step=0.1)
    asa_score = st.selectbox("ASA score", [1, 2, 3, 4], index=max(0, int(demo["asa_score"]) - 1))
    diabetes_status = st.checkbox("Diabetes", value=bool(demo["diabetes_status"]))
    nrs_2002_score = st.slider("NRS-2002 nutrition risk", 0.0, 7.0, float(demo["nrs_2002_score"]), step=0.5)

    st.divider()
    st.subheader("Current vitals")
    vitals = {
        "heart_rate": st.slider("Heart rate (bpm)", 45, 150, int(demo["vitals"]["heart_rate"])),
        "temperature": st.slider("Temperature (°C)", 35.0, 41.0, float(demo["vitals"]["temperature"]), step=0.1),
        "spo2": st.slider("SpO₂ (%)", 85, 100, int(demo["vitals"]["spo2"])),
    }

    st.divider()
    st.subheader("Current labs")
    labs = {
        "CRP": st.slider("CRP (mg/L)", 0, 220, int(demo["labs"]["CRP"])),
        "WBC": st.slider("WBC (cells/µL)", 3000, 25000, int(demo["labs"]["WBC"]), step=100),
        "albumin": st.slider("Albumin (g/dL)", 1.8, 5.0, float(demo["labs"]["albumin"]), step=0.1),
        "prealbumin": st.slider("Prealbumin (mg/L)", 40, 260, int(demo["labs"]["prealbumin"])),
        "glucose": st.slider("Glucose (mg/dL)", 70, 280, int(demo["labs"]["glucose"])),
        "zinc": st.slider("Serum zinc", 30, 130, int(demo["labs"]["zinc"])),
    }

    st.divider()
    st.subheader("Previous-24-hour nutrition intake")
    st.caption(
        "Layer 3 scores the phase-specific wound-healing nutrients and keeps the remaining meal totals "
        "for context, macro pattern selection, and recipe review."
    )

    n0 = demo["nutrition"]
    with st.expander("Primary wound-healing nutrition inputs", expanded=True):
        nutrition_primary = {
            "calories_kcal": st.slider("Total calories yesterday (kcal)", 0, 3500, int(n0.get("calories_kcal", 0)), step=25),
            "protein_g": st.slider("Total protein yesterday (g)", 0.0, 220.0, float(n0.get("protein_g", 0)), step=1.0),
            "carbohydrates_g": st.slider("Total carbohydrates yesterday (g)", 0.0, 500.0, float(n0.get("carbohydrates_g", 0)), step=5.0),
            "vitamin_c_mg": st.slider("Vitamin C yesterday (mg)", 0, 1500, int(n0.get("vitamin_c_mg", 0)), step=10),
            "hydration_ml": st.slider("Hydration yesterday (ml)", 0, 5000, int(n0.get("hydration_ml", 0)), step=50),
        }

    with st.expander("Additional meal totals tracked for context and recipes", expanded=False):
        nutrition_extra = {
            "total_fat_g": st.slider("Total fat yesterday (g)", 0.0, 200.0, float(n0.get("total_fat_g", 0)), step=1.0),
            "calcium_mg": st.slider("Calcium yesterday (mg)", 0, 2000, int(n0.get("calcium_mg", 0)), step=25),
            "cholesterol_mg": st.slider("Cholesterol yesterday (mg)", 0, 800, int(n0.get("cholesterol_mg", 0)), step=10),
            "iron_mg": st.slider("Iron yesterday (mg)", 0.0, 60.0, float(n0.get("iron_mg", 0)), step=0.5),
            "potassium_mg": st.slider("Potassium yesterday (mg)", 0, 6000, int(n0.get("potassium_mg", 0)), step=50),
            "saturated_fat_g": st.slider("Saturated fat yesterday (g)", 0.0, 80.0, float(n0.get("saturated_fat_g", 0)), step=0.5),
            "sodium_mg": st.slider("Sodium yesterday (mg)", 0, 6000, int(n0.get("sodium_mg", 0)), step=50),
            "sugar_g": st.slider("Sugar yesterday (g)", 0.0, 200.0, float(n0.get("sugar_g", 0)), step=1.0),
            "vitamin_d_ug": st.slider("Vitamin D yesterday (µg)", 0.0, 100.0, float(n0.get("vitamin_d_ug", 0)), step=0.5),
            "vitamin_e_mg": st.slider("Vitamin E yesterday (mg)", 0.0, 100.0, float(n0.get("vitamin_e_mg", 0)), step=0.5),
            "total_nutrient_value": st.slider("Total nutrient value", 0.0, 10000.0, float(n0.get("total_nutrient_value", 0)), step=10.0),
        }
    nutrition = {**nutrition_primary, **nutrition_extra}

overrides = {
    "patient_id": patient_id, "age": age, "sex": sex, "surgery_type": surgery_type, "day_post_op": day_post_op,
    "weight_kg": weight_kg, "bmi": bmi, "asa_score": asa_score, "diabetes_status": diabetes_status,
    "nrs_2002_score": nrs_2002_score, "vitals": vitals, "labs": labs, "nutrition": nutrition,
}
base_state = build_phase1_state(demo, overrides, scenario_key)

pipeline = run_three_layer_pipeline(base_state)
l1, l2, l3 = pipeline["layer1"], pipeline["layer2"], pipeline["layer3"]
phase = l1.get("current_phase")
priority = l3.get("overall_alert_priority") or l2.get("overall_alert_priority")

# ─── Overview ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 👤 Current Patient Snapshot")
profile_cols = st.columns(6)
expected_phase = demo.get("expected_phase")
with profile_cols[0]: st.metric("Patient", str(pipeline["state_after_layer3"].get("patient_id")))
with profile_cols[1]: st.metric("Day post-op", format_post_op_day(pipeline["state_after_layer3"]["phase1_input"].get("day_post_op")))
with profile_cols[2]: st.metric("Expected walkthrough", f"{phase_icon(str(expected_phase))} {expected_phase}" if expected_phase else "Uploaded JSON")
with profile_cols[3]: st.metric("Estimated phase", f"{phase_icon(str(phase))} {phase}")
with profile_cols[4]: st.metric("Layer 3 NSS", l3.get("nutritional_sufficiency_score"))
with profile_cols[5]: st.metric("Overall priority", f"{priority_icon(priority)} {priority}")

if expected_phase and str(phase).lower() != str(expected_phase).lower():
    st.warning(f"The selected walkthrough is **{expected_phase}**, but current slider values make Layer 1 estimate **{phase}**. This is allowed for experimentation; reset the scenario values for the clean walkthrough.")
else:
    st.success("Walkthrough and Layer 1 phase estimate are aligned.")

st.markdown("### 🧭 How the three layers connect")
flow_cols = st.columns([1, 0.16, 1, 0.16, 1])
with flow_cols[0]:
    st.container(border=True).markdown(f"**Layer 1 — HMM phase estimator**\n\nInput: day, vitals, labs, pain/appetite + trend-aware observed signals.\n\nOutput: **{phase_icon(str(phase))} {phase}** with confidence **{l1.get('phase_confidence')}**.")
with flow_cols[1]: st.markdown("### ➜")
with flow_cols[2]:
    st.container(border=True).markdown(f"**Layer 2 — Deviation detector**\n\nInput: Layer 1 phase + signal history.\n\nOutput: **{len(l2.get('deviation_flags', []))} deviation flags**, priority **{l2.get('overall_alert_priority')}**.")
with flow_cols[3]: st.markdown("### ➜")
with flow_cols[4]:
    st.container(border=True).markdown(f"**Layer 3 — Nutrition gap engine**\n\nInput: Layer 1 phase + Layer 2 flags + previous-24-hour meal intake.\n\nOutput: NSS **{l3.get('nutritional_sufficiency_score')}**, priority **{l3.get('overall_alert_priority')}**.")

st.markdown("### ⏱️ Biological wound-healing timeline")
phase_cols = st.columns(4)
for idx, ph in enumerate(PHASE_TIMELINE):
    active = str(phase).lower() == ph["phase"]
    with phase_cols[idx]:
        box = st.container(border=True)
        marker = "✅ Current estimate" if active else ""
        box.markdown(f"**{phase_icon(ph['phase'])} {ph['phase'].title()}**  \n_{ph['window']}_  \n{ph['purpose']}  \n{marker}")

# ─── Layer 1 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 1️⃣ Layer 1 — Healing Phase Estimator")
st.caption("Purpose: infer the hidden biological wound-healing phase from observable signals. This is not a diagnosis; it is a monitoring estimate.")
post = l1.get("phase_posteriors", {}) or {}
post_df = pd.DataFrame([{"phase": k, "posterior_probability": float(v)} for k, v in post.items()]).sort_values("posterior_probability", ascending=False)
c1, c2 = st.columns([1, 1])
with c1:
    st.metric("Most likely phase", f"{phase_icon(str(phase))} {phase}", f"confidence {l1.get('phase_confidence')}")
    for _, row in post_df.iterrows():
        st.write(f"{phase_icon(row['phase'])} **{row['phase']}** — {row['posterior_probability']:.1%}")
        st.progress(min(max(row["posterior_probability"], 0.0), 1.0))
with c2:
    feat_df = pd.DataFrame([{"feature": k, "value": v} for k, v in (l1.get("hmm_features_used", {}) or {}).items()])
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

st.markdown("### Four phase states considered by Layer 1")
phase_doc_cols = st.columns(4)
for idx, row in enumerate(PHASE_REFERENCE_ROWS):
    ph_key = str(row["Phase"]).lower()
    posterior_value = float((l1.get("phase_posteriors", {}) or {}).get(ph_key, 0.0))
    with phase_doc_cols[idx]:
        with st.container(border=True):
            st.markdown(f"**{phase_icon(ph_key)} {row['Phase']}**")
            st.caption(row["Duration"])
            st.progress(min(max(posterior_value, 0.0), 1.0))
            st.caption(f"Current posterior: {posterior_value:.1%}")

st.caption("Layer 1 shows all four possible hidden phase states with their current posterior probabilities. Extra phase definitions and raw output-contract JSON are hidden from the main demo to keep the flow clean.")

# ─── Layer 2 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2️⃣ Layer 2 — Deviation Detector")
st.caption("Purpose: compare the patient's observed trajectory with the expected trajectory for their clinical context. Change CRP, glucose, SpO₂, prealbumin, HR, temperature, WBC, albumin, or zinc in the sidebar and the flags below should change immediately.")
with st.expander("Layer 2 responsive rule map", expanded=False):
    st.dataframe(pd.DataFrame(DEVIATION_RULE_ROWS), hide_index=True, use_container_width=True)
    st.caption("Clinical deviation flags keep their gates: CRP after Day 3, albumin mainly during tissue-repair phases, and prealbumin from Day 5. Single-day outside-band observations are still reported below, but they do not automatically become escalation flags.")
flags = l2.get("deviation_flags", []) or []
dev_cols = st.columns(4)
with dev_cols[0]: st.metric("Layer 2 priority", f"{priority_icon(l2.get('overall_alert_priority'))} {l2.get('overall_alert_priority')}")
with dev_cols[1]: st.metric("Deviation flags", len(flags))
with dev_cols[2]: st.metric("Escalation required", "Yes" if l2.get("escalation_required") else "No")
with dev_cols[3]: st.metric("Signals monitored", len(l2.get("signal_predictions", {}) or {}))
if flags:
    for flag in flags:
        with st.container(border=True):
            st.markdown(f"**{priority_icon(flag.get('alert_priority'))} {flag.get('deviation_type', 'deviation')}** — {flag.get('alert_priority')}")
            st.write(flag.get("clinical_interpretation", ""))
            if flag.get("evidence"):
                st.caption("Evidence: " + " · ".join(map(str, flag.get("evidence", []))))
else:
    st.success("No clinically gated Layer 2 deviation flags generated for this run.")

single_day_reports = l2.get("single_day_reports", []) or []
if single_day_reports:
    with st.expander("Single-day observations reported by Layer 2", expanded=True):
        st.caption("These are outside the expected band on the current day. They are reported for physician awareness, but they do not count as deviation flags unless the clinical gate is met.")
        st.dataframe(pd.DataFrame(single_day_reports), hide_index=True, use_container_width=True)

pred_rows = []
for signal, pred in (l2.get("signal_predictions", {}) or {}).items():
    ci = pred.get("credible_interval_95", {}) or {}
    pred_rows.append({"signal": signal, "observed": clean_float(pred.get("observed_value"), 2), "expected": clean_float(pred.get("predicted_value"), 2), "95% lower": clean_float(ci.get("lower"), 2), "95% upper": clean_float(ci.get("upper"), 2), "outside interval": bool(pred.get("outside_credible_interval", False))})
if pred_rows:
    with st.expander("Signal prediction table"):
        st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True)

# ─── Layer 3 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3️⃣ Layer 3 — Nutritional Gap Engine")
st.caption("Purpose: score actual previous-24-hour intake against the phase-specific wound-healing targets selected by Layer 1.")
intake = l3.get("previous_24h_intake", {}) or {}
gap_cols = st.columns(4)
with gap_cols[0]: st.metric("NSS", l3.get("nutritional_sufficiency_score"), l3.get("nutritional_sufficiency_status"))
with gap_cols[1]: st.metric("Intake period", "previous 24h")
with gap_cols[2]: st.metric("Scored nutrients", len(l3.get("scored_nutritional_gaps", []) or []))
with gap_cols[3]: st.metric("Interactions", len(l3.get("interaction_flags", []) or []))

# v012: make the nutrition result visibly reactive, not only hidden in tables.
dominant_gap = l3.get("dominant_nutritional_gap", {}) or {}
context_notes = l3.get("contextual_nutrition_observations", []) or []
react_cols = st.columns(4)
with react_cols[0]:
    st.metric("Dominant nutrition gap", dominant_gap.get("display_name") or "None")
with react_cols[1]:
    st.metric("Dominant gap %", dominant_gap.get("gap_pct"))
with react_cols[2]:
    st.metric("Dominant priority", f"{priority_icon(dominant_gap.get('priority'))} {dominant_gap.get('priority')}")
with react_cols[3]:
    st.metric("Context notes", len(context_notes))

st.markdown("**Primary nutrition values used by Layer 3**")
primary_cols = st.columns(5)
with primary_cols[0]:
    st.metric("Protein", f"{clean_float(intake.get('protein_g_per_kg'), 2)} g/kg", f"total {clean_float(intake.get('protein_g'), 1)} g")
with primary_cols[1]:
    st.metric("Calories", f"{clean_float(intake.get('calories_kcal_per_kg'), 1)} kcal/kg", f"total {clean_float(intake.get('calories_kcal'), 0)} kcal")
with primary_cols[2]:
    st.metric("Vitamin C", f"{clean_float(intake.get('vitamin_c_mg'), 0)} mg", "previous 24h")
with primary_cols[3]:
    st.metric("Hydration", f"{clean_float(intake.get('hydration_ml_per_kg'), 1)} ml/kg", f"total {clean_float(intake.get('hydration_ml'), 0)} ml")
with primary_cols[4]:
    st.metric("Carb share", f"{clean_float(intake.get('carbohydrate_percent_calories'), 1)}%", f"carbs {clean_float(intake.get('carbohydrates_g'), 0)} g")

st.caption(
    "The sidebar still lets you adjust all meal totals. The main Layer-3 UI shows only the primary wound-healing values: "
    "protein, calories, Vitamin C, hydration, and carbohydrate share. Extra fields such as fat, sugar, sodium, Vitamin D, and Vitamin E are kept in the background for recipe/context logic."
)

gap_rows = []
for g in l3.get("scored_nutritional_gaps", []) or []:
    actual = g.get("actual", {}) or {}; target = g.get("target", {}) or {}
    gap_rows.append({"nutrient": g.get("display_name") or g.get("nutrient"), "actual": actual.get("value"), "unit": actual.get("unit") or target.get("unit"), "target_min": target.get("target_min"), "target_max": target.get("target_max"), "gap_pct": g.get("gap_pct"), "priority": f"{priority_icon(g.get('priority'))} {g.get('priority')}", "status": g.get("status")})
if gap_rows:
    st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)

interactions = l3.get("interaction_flags", []) or []
st.markdown("### Combined trajectory + nutrition interaction flags")
if interactions:
    for inter in interactions:
        with st.container(border=True):
            st.markdown(f"**{priority_icon(inter.get('alert_priority'))} {inter.get('interaction_type')}** — {inter.get('alert_priority')}")
            st.write(inter.get("combined_interpretation", ""))
            st.info(inter.get("alert_text", ""))
else:
    st.success("No combined interaction flags in this run.")

# ─── Final clinical insight + patient-facing recipe recommendation ───────────
st.markdown("---")
st.markdown("## 🍽️ Final Layer-3 Output — clinical insight + patient-facing recipe")
st.caption(
    "One physician-review insight is prepared for the language layer, followed by patient-friendly recipe ideas selected from the current macro pattern. "
    "These must be checked with the physician/dietitian before any diet change."
)
recipe_pick = build_macro_recipe_recommendation(pipeline)
clinical_narrative = build_clinical_narrative(pipeline)

st.markdown("### Clinical insight for physician review")
st.info(recipe_pick["clinical_insight"])
with st.expander("Structured narrative that can be passed to the language layer", expanded=False):
    st.text_area("Review-only clinical narrative", clinical_narrative, height=180)

summary_cols = st.columns(3)
with summary_cols[0]:
    st.metric("Current phase", recipe_pick["phase"])
with summary_cols[1]:
    st.metric("Selected macro pattern", recipe_pick["band"])
with summary_cols[2]:
    st.metric("Selection logic", recipe_pick["status"].replace("_", " "))

st.markdown("### Macro decision summary")
dg = recipe_pick.get("dominant_gap") or {}
dominant_gap_text = f"{dg.get('display_name', 'No major nutrition gap')} ({dg.get('gap_pct', 0.0)}% gap, {dg.get('priority', 'NONE')})"
with st.container(border=True):
    st.markdown("**Macro snapshot**")
    st.info(recipe_pick["macro_line"])

    st.markdown("**Dominant Layer-3 gap**")
    if str(dg.get("priority", "NONE")).upper() in {"HIGH", "CRITICAL"}:
        st.error(dominant_gap_text)
    elif str(dg.get("priority", "NONE")).upper() == "MODERATE":
        st.warning(dominant_gap_text)
    elif str(dg.get("priority", "NONE")).upper() == "LOW":
        st.info(dominant_gap_text)
    else:
        st.success(dominant_gap_text)

    st.markdown("**Selection trigger**")
    st.info(recipe_pick["trigger"])

st.warning("Patient-facing recipes are demo suggestions only. They must be checked and approved by the physician/dietitian before use.")

st.markdown("### Patient-facing recipe ideas for this macro pattern")
recipe_cols = st.columns(min(2, max(1, len(recipe_pick["recipes"]))))
for idx, selected_recipe in enumerate(recipe_pick["recipes"]):
    with recipe_cols[idx % len(recipe_cols)]:
        with st.container(border=True):
            st.markdown(f"### {selected_recipe['title']}")
            st.caption(f"Suggested slot: {selected_recipe['meal_slot']}")
            ingredients = selected_recipe.get("ingredients") or []
            if ingredients:
                st.markdown("**Ingredients:** " + ", ".join(ingredients))
            st.write(selected_recipe["patient_text"])
            st.info(selected_recipe["review_reason"])
st.caption(recipe_pick["safety_note"])

# ─── Technical details ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔧 Technical details: exact JSON payloads and outputs"):
    tabs = st.tabs(["State after Layer 3", "Layer 1", "Layer 2", "Layer 3"])
    with tabs[0]: st.json(pipeline["state_after_layer3"])
    with tabs[1]: st.json(l1)
    with tabs[2]: st.json(l2)
    with tabs[3]: st.json(l3)

st.caption("Clinical safety note: this is a demo of monitoring alerts and prioritisation. Physician and dietitian review remain mandatory before any clinical action.")
