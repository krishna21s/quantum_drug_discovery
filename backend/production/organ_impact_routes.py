"""
Organ Impact Prediction Routes
================================
FastAPI endpoint for predicting organ-level drug targets and
adverse effects from a SMILES string.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["Organ Impact Prediction"])


# ── Pydantic Schemas ────────────────────────────────────────────────

class OrganImpactRequest(BaseModel):
    smiles: str


class OrganEffectSchema(BaseModel):
    name: str
    reason: str
    confidence: float


class OrganImpactResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    target_organs: List[OrganEffectSchema]
    side_effect_organs: List[OrganEffectSchema]
    drug_class: str
    mechanism_summary: str


# ── Endpoint ────────────────────────────────────────────────────────

@router.post("/organ-impact", response_model=OrganImpactResponse)
async def predict_organ_impact(body: OrganImpactRequest):
    """
    Predict organ-level therapeutic targets and adverse effects for a molecule.

    Uses RDKit substructure matching against curated pharmacophore/toxicophore
    SMARTS patterns combined with physicochemical property-based distribution rules.
    """
    from production.services.organ_impact_service import predict_organ_impact as _predict

    try:
        result = _predict(body.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Organ impact prediction failed for '{body.smiles}': {e}")
        raise HTTPException(status_code=500, detail=f"Organ impact prediction failed: {str(e)}")

    return OrganImpactResponse(
        smiles=result.smiles,
        canonical_smiles=result.canonical_smiles,
        target_organs=[
            OrganEffectSchema(name=o.name, reason=o.reason, confidence=o.confidence)
            for o in result.target_organs
        ],
        side_effect_organs=[
            OrganEffectSchema(name=o.name, reason=o.reason, confidence=o.confidence)
            for o in result.side_effect_organs
        ],
        drug_class=result.drug_class,
        mechanism_summary=result.mechanism_summary,
    )
