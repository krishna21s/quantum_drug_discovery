"""
API Routes — Experiment Persistence
=====================================
Save / list / retrieve completed experiments from the database.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import get_db
from database.models import Experiment

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


# ── Schemas ───────────────────────────────────────────────

class SaveExperimentRequest(BaseModel):
    pdb_id: str
    target_name: Optional[str] = None
    temperature: float = 1.0
    n_candidates: int = 20
    stress_factors: list[str] = []
    docking_engine: str = "autodock_vina"
    vqe_optimizer: str = "COBYLA"
    vqe_max_iterations: int = 100
    run_admet: bool = True
    generation_time_s: float = 0
    n_sampled: int = 0
    n_valid: int = 0
    candidates_json: list[dict] = []

class ExperimentSummary(BaseModel):
    id: int
    pdb_id: str
    target_name: Optional[str]
    temperature: float
    n_candidates: int
    docking_engine: str
    vqe_optimizer: str
    generation_time_s: Optional[float]
    created_at: datetime
    candidate_count: int

    class Config:
        from_attributes = True

class ExperimentDetail(BaseModel):
    id: int
    pdb_id: str
    target_name: Optional[str]
    temperature: float
    n_candidates: int
    stress_factors: Optional[list[str]]
    docking_engine: str
    vqe_optimizer: str
    vqe_max_iterations: int
    run_admet: bool
    generation_time_s: Optional[float]
    n_sampled: Optional[int]
    n_valid: Optional[int]
    candidates_json: list[dict]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────

@router.post("/save", response_model=dict)
async def save_experiment(body: SaveExperimentRequest, db: Session = Depends(get_db)):
    """Save a completed experiment to permanent storage."""
    exp = Experiment(
        pdb_id=body.pdb_id.upper(),
        target_name=body.target_name or body.pdb_id.upper(),
        temperature=body.temperature,
        n_candidates=body.n_candidates,
        stress_factors=body.stress_factors,
        docking_engine=body.docking_engine,
        vqe_optimizer=body.vqe_optimizer,
        vqe_max_iterations=body.vqe_max_iterations,
        run_admet=body.run_admet,
        generation_time_s=body.generation_time_s,
        n_sampled=body.n_sampled,
        n_valid=body.n_valid,
        candidates_json=body.candidates_json,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return {"id": exp.id, "message": "Experiment saved successfully"}


@router.get("/", response_model=list[ExperimentSummary])
async def list_experiments(db: Session = Depends(get_db)):
    """List all saved experiments (most recent first)."""
    exps = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [
        ExperimentSummary(
            id=e.id,
            pdb_id=e.pdb_id,
            target_name=e.target_name,
            temperature=e.temperature,
            n_candidates=e.n_candidates,
            docking_engine=e.docking_engine,
            vqe_optimizer=e.vqe_optimizer,
            generation_time_s=e.generation_time_s,
            created_at=e.created_at,
            candidate_count=len(e.candidates_json) if e.candidates_json else 0,
        )
        for e in exps
    ]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Get full experiment details including all candidates."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentDetail(
        id=exp.id,
        pdb_id=exp.pdb_id,
        target_name=exp.target_name,
        temperature=exp.temperature,
        n_candidates=exp.n_candidates,
        stress_factors=exp.stress_factors,
        docking_engine=exp.docking_engine,
        vqe_optimizer=exp.vqe_optimizer,
        vqe_max_iterations=exp.vqe_max_iterations,
        run_admet=exp.run_admet,
        generation_time_s=exp.generation_time_s,
        n_sampled=exp.n_sampled,
        n_valid=exp.n_valid,
        candidates_json=exp.candidates_json or [],
        created_at=exp.created_at,
    )
