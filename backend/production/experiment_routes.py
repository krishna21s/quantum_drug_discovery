"""
Experiment API Routes
======================
Endpoints for experiment configuration, including
LLM-powered auto-parameter setting.
"""

import os
import sys
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiment", tags=["Experiment"])


# ── Schemas ─────────────────────────────────────────────────

class AutoConfigRequest(BaseModel):
    pdb_id: str = Field(..., description="PDB ID of the target protein")


class AutoConfigResponse(BaseModel):
    pdb_id: str
    protein_name: str = ""
    disease: str = ""
    temperature: float = 1.0
    n_candidates: int = 20
    vqe_optimizer: str = "COBYLA"
    vqe_max_iterations: int = 100
    docking_engine: str = "autodock_vina"
    stress_factors: list[str] = []
    run_admet: bool = True
    reasoning: str = ""


# ── Routes ──────────────────────────────────────────────────

@router.post("/auto-configure", response_model=AutoConfigResponse)
async def auto_configure(body: AutoConfigRequest):
    """
    Use LLM (Llama 3.3 70B via Groq) to intelligently set experiment
    parameters based on protein target analysis.
    """
    from production.services.llm_service import auto_configure_experiment

    # Optionally load phi for enriched context
    phi = None
    try:
        _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _V4_DIR = os.path.join(_BACKEND_DIR, "construction_v4")
        sys.path.insert(0, _V4_DIR)
        from training.pocket_conditioner import PocketConditioner
        from config_v4 import V4_CHECKPOINT_DIR

        pc = PocketConditioner(str(V4_CHECKPOINT_DIR))
        phi = pc.load_or_compute(body.pdb_id)
    except Exception as e:
        logger.warning(f"Could not load phi for {body.pdb_id}: {e}")

    try:
        config = auto_configure_experiment(body.pdb_id, phi)
        return AutoConfigResponse(**config)
    except Exception as e:
        logger.error(f"Auto-configure failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-configure failed: {str(e)}")
