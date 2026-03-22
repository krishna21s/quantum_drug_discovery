"""
Refinement API Routes
======================
FastAPI endpoint for iterative molecular lead optimization.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Lead Optimization"])


# ── Request / Response schemas ──────────────────────────────

class RefineRequest(BaseModel):
    smiles: str
    max_steps: int = 5
    variants_per_step: int = 15
    preserve_scaffold: bool = True


class RefineResponse(BaseModel):
    original_smiles: str
    final_smiles: str
    trajectory: list[dict]
    total_steps: int
    total_improvement: float
    elapsed_seconds: float
    converged: bool


# ── Endpoint ────────────────────────────────────────────────

@router.post("/refine", response_model=RefineResponse)
async def refine_molecule(request: Request, body: RefineRequest):
    """
    Iteratively refine a molecule to improve its binding affinity,
    ADMET properties, and reduce toxicity.

    Returns the full trajectory of the optimization so the frontend
    can visualize the step-by-step evolution.
    """
    from production.refinement.environment import optimize_candidate

    # Get oracles from app state (may be None if not loaded)
    binding_oracle = getattr(request.app.state, "binding_oracle", None)
    toxicity_pipeline = getattr(request.app.state, "pipeline", None)

    try:
        result = optimize_candidate(
            smiles=body.smiles,
            max_steps=body.max_steps,
            variants_per_step=body.variants_per_step,
            preserve_scaffold=body.preserve_scaffold,
            binding_oracle=binding_oracle,
            toxicity_pipeline=toxicity_pipeline,
        )
    except Exception as e:
        logger.error(f"Refinement failed for {body.smiles}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Refinement failed: {str(e)}",
        )

    return RefineResponse(
        original_smiles=result.original_smiles,
        final_smiles=result.final_smiles,
        trajectory=result.trajectory,
        total_steps=result.total_steps,
        total_improvement=result.total_improvement,
        elapsed_seconds=result.elapsed_seconds,
        converged=result.converged,
    )
