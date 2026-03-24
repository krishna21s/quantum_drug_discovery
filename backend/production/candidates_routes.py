"""
API Routes — Drug Candidate Endpoints
=======================================
Serves pre-computed RL candidates from final_candidates.json.
"""

import os
import sys
import json

from fastapi import APIRouter, HTTPException, Request

from .candidates_schemas import (
    CandidateItem,
    CandidatesListResponse,
    GenerateRequest,
    GenerateResponse,
)

router = APIRouter(prefix="/api", tags=["Drug Candidates"])


def _load_candidates_json() -> dict:
    """Load pre-computed candidates from V4 checkpoints."""
    _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates_path = os.path.join(
        _BACKEND_DIR, "construction_v4", "checkpoints", "final_candidates.json"
    )
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    with open(candidates_path, "r") as f:
        return json.load(f)


# ── Endpoints ───────────────────────────────────────────────

@router.get("/candidates", response_model=CandidatesListResponse)
async def get_candidates(request: Request):
    """
    Get all pre-computed drug candidates from RL fine-tuning.

    Returns the top-50 diverse EGFR-targeted candidates ranked by
    predicted binding affinity (XGB pIC50 + QSVR pIC50).
    """
    candidates_data = getattr(request.app.state, "candidates_data", None)
    if candidates_data is None:
        raise HTTPException(
            status_code=503,
            detail="Candidates not loaded. Check if final_candidates.json exists.",
        )

    items = [CandidateItem(**c) for c in candidates_data["candidates"]]

    return CandidatesListResponse(
        target=candidates_data.get("target", "EGFR (PDB 1M17)"),
        n_rl_episodes=candidates_data.get("n_rl_episodes", 0),
        total_generated=candidates_data.get("total_generated", 0),
        final_reward=candidates_data.get("final_reward", 0),
        total_time_min=candidates_data.get("total_time_min", 0),
        candidates=items,
    )


@router.get("/candidates/{rank}", response_model=CandidateItem)
async def get_candidate_by_rank(request: Request, rank: int):
    """Get a single candidate by its rank (1-indexed)."""
    candidates_data = getattr(request.app.state, "candidates_data", None)
    if candidates_data is None:
        raise HTTPException(status_code=503, detail="Candidates not loaded")

    for c in candidates_data["candidates"]:
        if c["rank"] == rank:
            return CandidateItem(**c)

    raise HTTPException(status_code=404, detail=f"Candidate with rank {rank} not found")


@router.post("/candidates/generate", response_model=GenerateResponse)
async def generate_candidates(request: Request, body: GenerateRequest):
    """
    Generate NEW drug candidates on-demand.

    Uses the RL-trained ConditionedRNN to sample molecules conditioned on
    the target protein's binding pocket. Molecules are filtered for validity
    and drug-likeness, then scored with dual oracles (XGB + QSVR).

    - **pdb_id**: Target protein PDB ID (default: 1M17 = EGFR)
    - **n_candidates**: How many valid candidates to return
    - **temperature**: Higher = more diverse, lower = more conservative
    """
    generator = getattr(request.app.state, "generator", None)
    if generator is None or not generator.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Generator model not loaded. Ensure policy_egfr_rl.pt exists.",
        )

    try:
        result = generator.generate(
            pdb_id=body.pdb_id,
            n_candidates=body.n_candidates,
            temperature=body.temperature,
            max_mw=body.max_mw,
            stress_factors=body.stress_factors,
            docking_engine=body.docking_engine,
            run_admet=body.run_admet,
            vqe_optimizer=body.vqe_optimizer,
            vqe_max_iterations=body.vqe_max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    items = [CandidateItem(**c) for c in result["candidates"]]

    return GenerateResponse(
        target=result["target"],
        n_requested=result["n_requested"],
        n_sampled=result["n_sampled"],
        n_valid=result["n_valid"],
        temperature=result["temperature"],
        generation_time_s=result["generation_time_s"],
        stress_applied=result.get("stress_applied", []),
        docking_engine=result.get("docking_engine", "none"),
        vqe_optimizer=result.get("vqe_optimizer", "COBYLA"),
        vqe_max_iterations=result.get("vqe_max_iterations", 100),
        candidates=items,
    )
