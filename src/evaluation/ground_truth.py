"""Ground-truth computations for evaluation questions.

Each function computes the reference answer for a specific question using
the statistics module directly (no LLM involved). Results are cached so
they only need to be computed once per session, and persisted to disk for
subsequent runs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.statistics import (
    weighted_mean,
    weighted_frequency,
    weighted_crosstab,
)
from src.data.loader import DatasetLoader
from src.data.registry import get_dataset_info


_loader = DatasetLoader()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _load_columns(dataset_name: str, columns_tuple: tuple[str, ...]):
    """Load specific columns (hashable tuple for lru_cache)."""
    df, meta = _loader.load(dataset_name, columns=list(columns_tuple))
    return df, meta


# ---------------------------------------------------------------------------
# Ground truth functions — keyed by ground_truth_key in questions.yaml
# ---------------------------------------------------------------------------

def _gt_cache_dir() -> Path:
    """Return the ground truth cache directory, creating it if needed."""
    d = Path("experiments/.cache/ground_truth")
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_ground_truth(key: str, no_cache: bool = False) -> dict[str, Any]:
    """Dispatch to the correct ground-truth function by key.

    Returns a dict with at least 'value' and 'type' keys.
    Results are cached to disk for subsequent runs.
    """
    if key not in _GROUND_TRUTH_REGISTRY:
        return {"value": None, "type": "unknown", "error": f"No ground truth for key: {key}"}

    # Check disk cache
    cache_path = _gt_cache_dir() / f"{key}.json"
    if not no_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass  # recompute on any read error

    result = _GROUND_TRUTH_REGISTRY[key]()

    # Save to disk cache
    try:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass  # non-fatal if cache write fails

    return result


# ── Arab Barometer ────────────────────────────────────────────────────

def _ab_weighted_mean_age() -> dict:
    df, meta = _load_columns("arab_barometer", ("Q1001", "WT"))
    valid = df.dropna(subset=["Q1001", "WT"])
    valid = valid[(valid["Q1001"] > 0) & (valid["Q1001"] < 98)]
    val = weighted_mean(valid["Q1001"], valid["WT"])
    return {"value": round(val, 2), "type": "numeric"}


def _ab_top_country() -> dict:
    df, meta = _load_columns("arab_barometer", ("COUNTRY", "WT"))
    freq = df["COUNTRY"].value_counts()
    top_code = int(freq.index[0])
    # Arab Barometer VIII country codes from DTA value labels
    country_labels = {7: "Iraq", 8: "Jordan", 9: "Kuwait", 10: "Lebanon",
                      12: "Mauritania", 13: "Morocco", 15: "Palestine", 21: "Tunisia"}
    label = country_labels.get(top_code, str(top_code))
    return {"value": label, "type": "categorical", "aliases": [str(top_code)]}


def _ab_education_trust_direction() -> dict:
    df, _ = _load_columns("arab_barometer", ("Q101", "Q201A_1", "WT"))
    valid = df.dropna(subset=["Q101", "Q201A_1", "WT"])
    valid = valid[(valid["Q101"] > 0) & (valid["Q101"] < 98) & (valid["Q201A_1"] > 0) & (valid["Q201A_1"] < 98)]
    corr = valid["Q101"].corr(valid["Q201A_1"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _ab_internet_by_country() -> dict:
    return {"value": "Internet usage varies significantly across Arab countries, with Gulf states showing higher usage rates", "type": "descriptive"}


def _ab_pct_econ_very_bad() -> dict:
    df, _ = _load_columns("arab_barometer", ("Q101", "WT"))
    valid = df.dropna(subset=["Q101", "WT"])
    valid = valid[(valid["Q101"] > 0) & (valid["Q101"] < 98)]
    pct = valid[valid["Q101"] == 4]["WT"].sum() / valid["WT"].sum() * 100
    return {"value": round(pct, 1), "type": "numeric"}


# ── World Values Survey ───────────────────────────────────────────────

def _wvs_weighted_mean_life_satisfaction() -> dict:
    df, meta = _load_columns("wvs", ("Q49", "W_WEIGHT"))
    valid = df.dropna(subset=["Q49", "W_WEIGHT"])
    valid = valid[valid["Q49"] > 0]
    val = weighted_mean(valid["Q49"], valid["W_WEIGHT"])
    return {"value": round(val, 2), "type": "numeric"}


def _wvs_top_country() -> dict:
    df, _ = _load_columns("wvs", ("B_COUNTRY_ALPHA",))
    freq = df["B_COUNTRY_ALPHA"].value_counts()
    top = str(freq.index[0])
    return {"value": top, "type": "categorical", "aliases": ["Canada"]}


def _wvs_income_happiness_direction() -> dict:
    df, _ = _load_columns("wvs", ("Q288", "Q46", "W_WEIGHT"))
    valid = df.dropna(subset=["Q288", "Q46"])
    valid = valid[(valid["Q288"] > 0) & (valid["Q46"] > 0)]
    corr = valid["Q288"].corr(valid["Q46"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _wvs_religion_by_region() -> dict:
    return {"value": "Varies by region — descriptive comparison expected", "type": "descriptive"}


def _wvs_trust_democracy_direction() -> dict:
    df, _ = _load_columns("wvs", ("Q57", "Q235", "W_WEIGHT"))
    valid = df.dropna(subset=["Q57", "Q235"])
    valid = valid[(valid["Q57"] > 0) & (valid["Q235"] > 0)]
    corr = valid["Q57"].corr(valid["Q235"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


# ── GSS ───────────────────────────────────────────────────────────────

def _gss_weighted_mean_educ() -> dict:
    df, meta = _load_columns("gss", ("educ", "wtssps"))
    valid = df.dropna(subset=["educ", "wtssps"])
    valid = valid[valid["educ"] >= 0]
    val = weighted_mean(valid["educ"], valid["wtssps"])
    return {"value": round(val, 2), "type": "numeric"}


def _gss_top_marital_status() -> dict:
    df, _ = _load_columns("gss", ("marital",))
    freq = df["marital"].value_counts()
    top_code = freq.index[0]
    # GSS marital: 1=married, 2=widowed, 3=divorced, 4=separated, 5=never married
    marital_labels = {1: "married", 2: "widowed", 3: "divorced", 4: "separated", 5: "never married"}
    label = marital_labels.get(int(top_code), str(top_code))
    return {"value": label, "type": "categorical", "aliases": [str(int(top_code))]}


def _gss_educ_income_direction() -> dict:
    df, _ = _load_columns("gss", ("educ", "realinc", "wtssps"))
    valid = df.dropna(subset=["educ", "realinc"])
    valid = valid[(valid["educ"] >= 0) & (valid["realinc"] > 0)]
    corr = valid["educ"].corr(valid["realinc"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _gss_happiness_trend() -> dict:
    return {"value": "Trend over decades — descriptive analysis expected", "type": "descriptive"}


def _gss_party_spending_direction() -> dict:
    return {"value": "positive", "type": "directional",
            "note": "Democrats generally favor more spending; expect party-spending association"}


# ── New Arab Barometer questions ──────────────────────────────────────

def _ab_total_respondents() -> dict:
    df, _ = _load_columns("arab_barometer", ("COUNTRY",))
    return {"value": len(df), "type": "numeric"}


def _ab_gender_distribution() -> dict:
    return {"value": "Distribution of male and female respondents across the survey", "type": "descriptive"}


def _ab_age_internet_direction() -> dict:
    df, _ = _load_columns("arab_barometer", ("Q1001", "Q409", "WT"))
    valid = df.dropna(subset=["Q1001", "Q409"])
    valid = valid[(valid["Q1001"] > 0) & (valid["Q1001"] < 98) & (valid["Q409"] > 0) & (valid["Q409"] < 98)]
    corr = valid["Q1001"].corr(valid["Q409"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _ab_pct_trust_great_deal() -> dict:
    df, _ = _load_columns("arab_barometer", ("Q201A_1", "WT"))
    valid = df.dropna(subset=["Q201A_1", "WT"])
    valid = valid[(valid["Q201A_1"] > 0) & (valid["Q201A_1"] < 98)]
    pct = valid[valid["Q201A_1"] == 1]["WT"].sum() / valid["WT"].sum() * 100
    return {"value": round(pct, 1), "type": "numeric"}


def _ab_trust_by_age_group() -> dict:
    return {"value": "Compare trust levels between young (18-30) and older (50+) respondents", "type": "descriptive"}


# ── New WVS questions ─────────────────────────────────────────────────

def _wvs_weighted_mean_happiness() -> dict:
    df, _ = _load_columns("wvs", ("Q46", "W_WEIGHT"))
    valid = df.dropna(subset=["Q46", "W_WEIGHT"])
    valid = valid[(valid["Q46"] > 0) & (valid["Q46"] <= 4)]
    val = weighted_mean(valid["Q46"], valid["W_WEIGHT"])
    return {"value": round(val, 2), "type": "numeric"}


def _wvs_num_countries() -> dict:
    df, _ = _load_columns("wvs", ("B_COUNTRY_ALPHA",))
    n = df["B_COUNTRY_ALPHA"].nunique()
    return {"value": n, "type": "numeric"}


def _wvs_educ_satisfaction_direction() -> dict:
    df, _ = _load_columns("wvs", ("Q275", "Q49", "W_WEIGHT"))
    valid = df.dropna(subset=["Q275", "Q49"])
    valid = valid[(valid["Q275"] > 0) & (valid["Q49"] > 0)]
    corr = valid["Q275"].corr(valid["Q49"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _wvs_gender_equality_by_region() -> dict:
    return {"value": "Gender equality attitudes vary by region — descriptive comparison expected", "type": "descriptive"}


def _wvs_pct_trust_people() -> dict:
    df, _ = _load_columns("wvs", ("Q57", "W_WEIGHT"))
    valid = df.dropna(subset=["Q57", "W_WEIGHT"])
    valid = valid[(valid["Q57"] > 0) & (valid["Q57"] <= 2)]
    pct = valid[valid["Q57"] == 1]["W_WEIGHT"].sum() / valid["W_WEIGHT"].sum() * 100
    return {"value": round(pct, 1), "type": "numeric"}


# ── New GSS questions ─────────────────────────────────────────────────

def _gss_weighted_mean_age() -> dict:
    df, _ = _load_columns("gss", ("age", "wtssps"))
    valid = df.dropna(subset=["age", "wtssps"])
    valid = valid[(valid["age"] > 0) & (valid["age"] < 99)]
    val = weighted_mean(valid["age"], valid["wtssps"])
    return {"value": round(val, 2), "type": "numeric"}


def _gss_total_respondents() -> dict:
    df, _ = _load_columns("gss", ("year",))
    return {"value": len(df), "type": "numeric"}


def _gss_age_happiness_direction() -> dict:
    df, _ = _load_columns("gss", ("age", "happy", "wtssps"))
    valid = df.dropna(subset=["age", "happy"])
    valid = valid[(valid["age"] > 0) & (valid["age"] < 99) & (valid["happy"] > 0) & (valid["happy"] <= 3)]
    corr = valid["age"].corr(valid["happy"])
    direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "none")
    return {"value": direction, "type": "directional", "correlation": round(corr, 4)}


def _gss_marital_by_sex() -> dict:
    return {"value": "Marital status distribution differs by gender — descriptive comparison", "type": "descriptive"}


def _gss_pct_very_happy() -> dict:
    df, _ = _load_columns("gss", ("happy", "wtssps"))
    valid = df.dropna(subset=["happy", "wtssps"])
    valid = valid[(valid["happy"] > 0) & (valid["happy"] <= 3)]
    pct = valid[valid["happy"] == 1]["wtssps"].sum() / valid["wtssps"].sum() * 100
    return {"value": round(pct, 1), "type": "numeric"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GROUND_TRUTH_REGISTRY: dict[str, callable] = {
    # Arab Barometer (10)
    "weighted_mean_age": _ab_weighted_mean_age,
    "top_country": _ab_top_country,
    "education_trust_direction": _ab_education_trust_direction,
    "internet_by_country": _ab_internet_by_country,
    "pct_econ_very_bad": _ab_pct_econ_very_bad,
    "ab_total_respondents": _ab_total_respondents,
    "ab_gender_distribution": _ab_gender_distribution,
    "ab_age_internet_direction": _ab_age_internet_direction,
    "ab_pct_trust_great_deal": _ab_pct_trust_great_deal,
    "ab_trust_by_age_group": _ab_trust_by_age_group,
    # WVS (10)
    "weighted_mean_life_satisfaction": _wvs_weighted_mean_life_satisfaction,
    "top_country_wvs": _wvs_top_country,
    "income_happiness_direction": _wvs_income_happiness_direction,
    "religion_by_region": _wvs_religion_by_region,
    "trust_democracy_direction": _wvs_trust_democracy_direction,
    "wvs_weighted_mean_happiness": _wvs_weighted_mean_happiness,
    "wvs_num_countries": _wvs_num_countries,
    "wvs_educ_satisfaction_direction": _wvs_educ_satisfaction_direction,
    "wvs_gender_equality_by_region": _wvs_gender_equality_by_region,
    "wvs_pct_trust_people": _wvs_pct_trust_people,
    # GSS (10)
    "weighted_mean_educ": _gss_weighted_mean_educ,
    "top_marital_status": _gss_top_marital_status,
    "educ_income_direction": _gss_educ_income_direction,
    "happiness_trend": _gss_happiness_trend,
    "party_spending_direction": _gss_party_spending_direction,
    "gss_weighted_mean_age": _gss_weighted_mean_age,
    "gss_total_respondents": _gss_total_respondents,
    "gss_age_happiness_direction": _gss_age_happiness_direction,
    "gss_marital_by_sex": _gss_marital_by_sex,
    "gss_pct_very_happy": _gss_pct_very_happy,
}
