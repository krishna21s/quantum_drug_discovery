"""
API Routes — Toxicity Prediction Endpoints
============================================
Production FastAPI router with health, single, and batch prediction.
"""

import time
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from rdkit import Chem

from .schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    TimingInfo,
    ConfidenceInterval,
)

router = APIRouter(prefix="/api", tags=["Toxicity Prediction"])


# ── Helpers ─────────────────────────────────────────────────────────

def _validate_smiles(smiles: str) -> str:
    """Validate and canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SMILES string: '{smiles}'. Could not be parsed by RDKit.",
        )
    return Chem.MolToSmiles(mol, canonical=True)


def _build_response(result: dict, canonical: str) -> PredictResponse:
    """Convert orchestrator result dict to typed response."""
    ensemble_prob = result["ensemble_prob"]
    is_toxic = ensemble_prob > 0.5

    timings = TimingInfo(
        xgb_ms=result["timings"].get("xgb_ms", 0),
        quantum_s=result["timings"].get("quantum_s", 0),
        total_s=sum(
            v for v in result["timings"].values()
            if isinstance(v, (int, float))
        ),
        shot_s=result["timings"].get("shot_s"),
    )

    ci = None
    if result.get("mode") == "full":
        ci = ConfidenceInterval(
            probability=result.get("quantum_prob_shot", result["quantum_prob"]),
            std=result.get("quantum_std", 0),
            ci_lower=result.get("quantum_ci_lower", result["quantum_prob"]),
            ci_upper=result.get("quantum_ci_upper", result["quantum_prob"]),
            n_bootstrap=result.get("n_bootstrap", 0),
        )

    return PredictResponse(
        smiles=result["smiles"],
        canonical_smiles=canonical,
        classical_probability=round(result["xgb_prob"], 6),
        quantum_probability=round(result["quantum_prob"], 6),
        ensemble_probability=round(ensemble_prob, 6),
        baseline_score=round(result.get("baseline_score", 0), 6),
        verdict="HIGH TOXICITY RISK" if is_toxic else "LOW TOXICITY RISK",
        confidence=round(ensemble_prob if is_toxic else (1 - ensemble_prob), 4),
        timings=timings,
        quantum_cached=result.get("quantum_cached", False),
        ci=ci,
        mode=result.get("mode", "fast"),
    )


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check pipeline health and loaded models."""
    pipeline = getattr(request.app.state, "pipeline", None)
    feature_svc = getattr(request.app.state, "feature_svc", None)

    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            models_loaded=[],
            pipeline_ready=False,
            checkpoint_dir="",
        )

    models = []
    if pipeline.classical and pipeline.classical.xgb_model:
        models.append("xgboost")
    if pipeline.quantum:
        models.append("quantum_kernel_svm")

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "construction_v2",
    ))
    from config import CHECKPOINT_DIR

    return HealthResponse(
        status="healthy",
        models_loaded=models,
        pipeline_ready=True,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_single(request: Request, body: PredictRequest):
    """
    Predict toxicity for a single molecule.

    - **Fast mode** (default): XGBoost + statevector quantum (≤3s)
    - **Full mode** (enable_ci=True): Adds shot-based CI (15-120s)
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    canonical = _validate_smiles(body.smiles)

    try:
        if body.enable_ci:
            result = pipeline.predict_full(
                body.smiles,
                n_bootstrap=body.n_bootstrap,
            )
        else:
            result = pipeline.predict_fast(body.smiles)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    return _build_response(result, canonical)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: Request, body: BatchPredictRequest):
    """
    Predict toxicity for a batch of molecules (max 50).

    All predictions use fast mode (statevector quantum).
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    # Validate all SMILES first
    canonicals = {}
    for smiles in body.smiles_list:
        canonicals[smiles] = _validate_smiles(smiles)

    t0 = time.time()
    predictions = []
    errors = []

    for smiles in body.smiles_list:
        try:
            result = pipeline.predict_fast(smiles)
            predictions.append(_build_response(result, canonicals[smiles]))
        except Exception as e:
            errors.append({"smiles": smiles, "error": str(e)})

    total_time = time.time() - t0

    # Summary statistics
    if predictions:
        ensemble_probs = [p.ensemble_probability for p in predictions]
        summary = {
            "total_molecules": len(body.smiles_list),
            "successful": len(predictions),
            "failed": len(errors),
            "avg_ensemble_probability": round(float(np.mean(ensemble_probs)), 4),
            "high_risk_count": sum(1 for p in ensemble_probs if p > 0.5),
            "low_risk_count": sum(1 for p in ensemble_probs if p <= 0.5),
        }
        if errors:
            summary["errors"] = errors
    else:
        summary = {
            "total_molecules": len(body.smiles_list),
            "successful": 0,
            "failed": len(errors),
            "errors": errors,
        }

    return BatchPredictResponse(
        predictions=predictions,
        summary=summary,
        total_time_s=round(total_time, 3),
    )
