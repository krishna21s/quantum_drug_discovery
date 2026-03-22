"""
ADMET Prediction Service
========================
Uses the open-source ADMET-AI library (https://github.com/swansonk14/admet_ai)
to generate real ADMET property predictions from SMILES strings.

The raw ADMET-AI output contains dozens of individual endpoints.
This service maps them into our normalized DB schema:
    absorption  (0-1)
    distribution (0-1)
    metabolism   (0-1)
    excretion    (0-1)
    overall      (0-1)
    verdict      ("Pass" | "Caution" | "Fail")
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ── Lazy model singleton ────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    """Load ADMETModel once and cache it for the process lifetime."""
    from admet_ai import ADMETModel  # heavy import, deferred
    logger.info("Loading ADMET-AI model (first call, may download weights)...")
    model = ADMETModel()
    logger.info("ADMET-AI model loaded successfully.")
    return model


# ── Helper: clamp to [0, 1] ─────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Core prediction function ────────────────────────────────

def generate_admet_from_smiles(smiles: str) -> dict[str, Any]:
    """
    Accept a SMILES string, run ADMET-AI, and return a dict compatible
    with our database ADMET schema:
        {
            "absorption": float,
            "distribution": float,
            "metabolism": float,
            "excretion": float,
            "overall": float,
            "verdict": str,
            "raw": { ... all ADMET-AI predictions ... }
        }
    """
    model = _get_model()
    raw: dict = model.predict(smiles=smiles)

    # ── Map raw outputs → category scores ────────────────────
    absorption = _score_absorption(raw)
    distribution = _score_distribution(raw)
    metabolism = _score_metabolism(raw)
    excretion = _score_excretion(raw)

    overall = _clamp(
        absorption * 0.25
        + distribution * 0.15
        + metabolism * 0.20
        + excretion * 0.15
        + _score_safety(raw) * 0.25
    )

    verdict = "Pass" if overall > 0.70 else "Caution" if overall > 0.45 else "Fail"

    return {
        "absorption": round(absorption, 4),
        "distribution": round(distribution, 4),
        "metabolism": round(metabolism, 4),
        "excretion": round(excretion, 4),
        "overall": round(overall, 4),
        "verdict": verdict,
        "raw": raw,
    }


# ── Category scoring helpers ────────────────────────────────
# Each helper inspects the raw dict for relevant ADMET-AI keys.
# If a key is absent we fall back to a neutral 0.5 score.

def _safe_get(raw: dict, key: str, default: float = 0.5) -> float:
    """Safely get a float value from the raw predictions."""
    val = raw.get(key, default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _score_absorption(raw: dict) -> float:
    """
    Derives absorption score from:
    - HIA (human intestinal absorption) — higher is better
    - Caco-2 permeability — higher is better
    - Bioavailability — higher is better
    - Solubility — higher is better (logS)
    """
    scores = []

    # HIA_Hou — binary classification (probability of positive)
    hia = _safe_get(raw, "HIA_Hou", 0.5)
    scores.append(hia)

    # Caco-2 permeability (log cm/s) — typical range: -7 to -4
    # We normalize: > -5.15 is good (score ~1), < -6 is poor (score ~0)
    caco2 = _safe_get(raw, "Caco2_Wang", -5.5)
    caco2_score = _clamp((caco2 + 7) / 3)  # maps [-7, -4] → [0, 1]
    scores.append(caco2_score)

    # Bioavailability (Ma) — probability
    bioavail = _safe_get(raw, "Bioavailability_Ma", 0.5)
    scores.append(bioavail)

    # Solubility (ESOL, logS) — higher is more soluble, typical: -6 to 0
    sol = _safe_get(raw, "Solubility_AqSolDB", -3.0)
    sol_score = _clamp((sol + 6) / 6)  # maps [-6, 0] → [0, 1]
    scores.append(sol_score)

    return _clamp(sum(scores) / len(scores)) if scores else 0.5


def _score_distribution(raw: dict) -> float:
    """
    Derives distribution score from:
    - BBB permeability — binary (higher prob = better distribution)
    - VDss — volume of distribution
    - Plasma protein binding (PPBR)
    """
    scores = []

    # BBB permeability (Martins) — probability
    bbb = _safe_get(raw, "BBB_Martins", 0.5)
    scores.append(bbb)

    # VDss (Lombardo) — log L/kg, typical range: -1 to 2
    vdss = _safe_get(raw, "VDss_Lombardo", 0.3)
    # Optimal range: 0.04-20 L/kg (log: -1.4 to 1.3)
    vdss_score = _clamp(1 - abs(vdss - 0.3) / 2)  # bell curve around 0.3
    scores.append(vdss_score)

    # PPBR (%) — too high (>95%) means poor free fraction
    ppbr = _safe_get(raw, "PPBR_AZ", 80.0)
    ppbr_score = _clamp(1 - (ppbr / 100))  # lower binding = better score
    ppbr_score = max(ppbr_score, 0.2)  # floor at 0.2
    scores.append(ppbr_score)

    return _clamp(sum(scores) / len(scores)) if scores else 0.5


def _score_metabolism(raw: dict) -> float:
    """
    Derives metabolism score from CYP450 inhibition predictions.
    Being a CYP inhibitor is undesirable (drug-drug interactions).
    """
    cyp_keys = [
        "CYP2C19_Veith",
        "CYP2D6_Veith",
        "CYP3A4_Veith",
        "CYP1A2_Veith",
        "CYP2C9_Veith",
    ]

    inhibitor_count = 0
    found = 0
    for key in cyp_keys:
        val = raw.get(key)
        if val is not None:
            found += 1
            try:
                if float(val) > 0.5:
                    inhibitor_count += 1
            except (ValueError, TypeError):
                pass

    if found == 0:
        return 0.5

    # Fewer CYP inhibitions = better score
    ratio = inhibitor_count / found
    return _clamp(1 - ratio)


def _score_excretion(raw: dict) -> float:
    """
    Derives excretion score from:
    - Half-life (Obach) — moderate half-life is preferable
    - Clearance — moderate clearance is ideal
    """
    scores = []

    # Half-life (Obach) — binary: long half-life (1) vs short (0)
    hl = _safe_get(raw, "Half_Life_Obach", 0.5)
    # Moderate half-life is ideal, so we give best score around 0.5
    hl_score = _clamp(1 - abs(hl - 0.5) * 2)
    scores.append(max(hl_score, 0.3))

    # Clearance — hepatocyte / microsome
    cl_h = _safe_get(raw, "Clearance_Hepatocyte_AZ", 50.0)
    # Optimal clearance: 10-70 μL/min/10^6 cells
    cl_score = _clamp(1 - abs(cl_h - 40) / 80)
    scores.append(max(cl_score, 0.2))

    cl_m = _safe_get(raw, "Clearance_Microsome_AZ", 50.0)
    cl_m_score = _clamp(1 - abs(cl_m - 40) / 80)
    scores.append(max(cl_m_score, 0.2))

    return _clamp(sum(scores) / len(scores)) if scores else 0.5


def _score_safety(raw: dict) -> float:
    """
    Derives a safety/toxicity score from ADMET-AI toxicity endpoints.
    Higher = safer (inversely related to toxicity).
    """
    tox_keys = [
        "hERG",
        "AMES",
        "DILI",
        "Skin_Reaction",
        "Carcinogens_Lagunin",
        "ClinTox",
    ]

    toxic_count = 0
    found = 0
    for key in tox_keys:
        val = raw.get(key)
        if val is not None:
            found += 1
            try:
                if float(val) > 0.5:  # predicted as toxic/positive
                    toxic_count += 1
            except (ValueError, TypeError):
                pass

    if found == 0:
        return 0.5

    # Fewer positive toxicity flags = safer
    ratio = toxic_count / found
    return _clamp(1 - ratio)
