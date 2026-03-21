"""
API Routes — Binding Affinity Scoring
=======================================
Live dual-oracle (XGB + QSVR) binding affinity prediction.
"""

import time
from fastapi import APIRouter, HTTPException, Request

from rdkit import Chem

from .candidates_schemas import (
    BindingScoreRequest,
    BindingScoreResponse,
    BatchBindingRequest,
    BatchBindingResponse,
)

router = APIRouter(prefix="/api", tags=["Binding Affinity"])


def _validate_smiles(smiles: str) -> str:
    """Validate and canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SMILES string: '{smiles}'",
        )
    return Chem.MolToSmiles(mol, canonical=True)


@router.post("/binding/score", response_model=BindingScoreResponse)
async def score_binding(request: Request, body: BindingScoreRequest):
    """
    Score a single molecule for binding affinity (pIC50).

    Uses dual oracles:
    - **XGBoost**: 4273-d fingerprint regression (fast, R²≈0.6)
    - **QSVR**: Quantum kernel SVR with Nyström approximation (R²≈0.27)

    Higher pIC50 = stronger binding. pIC50 > 6.0 suggests promising activity.
    """
    binding_oracle = getattr(request.app.state, "binding_oracle", None)
    if binding_oracle is None:
        raise HTTPException(
            status_code=503,
            detail="Binding affinity oracle not loaded",
        )

    canonical = _validate_smiles(body.smiles)

    try:
        result = binding_oracle.score(body.smiles)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}",
        )

    return BindingScoreResponse(
        smiles=body.smiles,
        canonical_smiles=canonical,
        xgb_pic50=result.get("xgb_pic50"),
        qsvr_pic50=result.get("pic50"),
        scoring_mode=result.get("mode", "unknown"),
        latency_s=result.get("latency_s", 0),
        error=result.get("error"),
    )


@router.post("/binding/score/batch", response_model=BatchBindingResponse)
async def score_binding_batch(request: Request, body: BatchBindingRequest):
    """
    Score a batch of molecules for binding affinity (max 50).

    All predictions use both XGB and QSVR oracles.
    """
    binding_oracle = getattr(request.app.state, "binding_oracle", None)
    if binding_oracle is None:
        raise HTTPException(
            status_code=503,
            detail="Binding affinity oracle not loaded",
        )

    t0 = time.time()
    predictions = []
    errors = []

    for smiles in body.smiles_list:
        try:
            canonical = _validate_smiles(smiles)
            result = binding_oracle.score(smiles)
            predictions.append(BindingScoreResponse(
                smiles=smiles,
                canonical_smiles=canonical,
                xgb_pic50=result.get("xgb_pic50"),
                qsvr_pic50=result.get("pic50"),
                scoring_mode=result.get("mode", "unknown"),
                latency_s=result.get("latency_s", 0),
                error=result.get("error"),
            ))
        except Exception as e:
            errors.append({"smiles": smiles, "error": str(e)})

    total_time = time.time() - t0

    xgb_scores = [p.xgb_pic50 for p in predictions if p.xgb_pic50 is not None]
    qsvr_scores = [p.qsvr_pic50 for p in predictions if p.qsvr_pic50 is not None]

    summary = {
        "total_molecules": len(body.smiles_list),
        "successful": len(predictions),
        "failed": len(errors),
        "avg_xgb_pic50": round(sum(xgb_scores) / len(xgb_scores), 3) if xgb_scores else None,
        "avg_qsvr_pic50": round(sum(qsvr_scores) / len(qsvr_scores), 3) if qsvr_scores else None,
    }
    if errors:
        summary["errors"] = errors

    return BatchBindingResponse(
        predictions=predictions,
        summary=summary,
        total_time_s=round(total_time, 3),
    )
