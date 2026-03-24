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
from database.models import Experiment, Candidate, BindingAffinity, Toxicity, ADMET

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
    
    # ── Unpack Candidates & prevent duplicates ──
    added_count = 0
    for c_data in body.candidates_json:
        smiles = c_data.get("smiles")
        if not smiles:
            continue
            
        # Check if exists
        existing = db.query(Candidate).filter(Candidate.smiles == smiles).first()
        if existing:
            continue
            
        # 1. Create Candidate
        target_val = body.target_name or body.pdb_id.upper()
        new_candidate = Candidate(
            smiles=smiles,
            target=target_val,
            mw=c_data.get("mw"),
            logp=c_data.get("logp"),
            tpsa=c_data.get("tpsa"),
            qed=c_data.get("qed"),
            sa_score=c_data.get("sa_score"),
            lipinski_pass=c_data.get("lipinski_pass", False),
            is_novel=c_data.get("is_novel", True)
        )
        db.add(new_candidate)
        db.flush() # Flush to get new_candidate.id
        
        # 2. Add Binding Affinity
        xgb_score = c_data.get("xgb_pic50")
        quantum_score = c_data.get("quantum_pic50")
        if xgb_score is not None or quantum_score is not None:
            binding = BindingAffinity(
                candidate_id=new_candidate.id,
                xgb_pic50=xgb_score,
                qsvr_pic50=quantum_score,
                scoring_mode="qsvr_rbf"
            )
            db.add(binding)
            
        # 3. Add Toxicity
        tox_data = c_data.get("toxicity")
        if tox_data is not None:
            tox = Toxicity(
                candidate_id=new_candidate.id,
                canonical_smiles=smiles,
                toxicity_score=tox_data.get("toxicity_score"),
                is_toxic=tox_data.get("is_toxic"),
                alerts_json=tox_data.get("alerts", [])
            )
            db.add(tox)
            
        # 4. Add ADMET
        admet_data = c_data.get("admet")
        if admet_data is not None:
            admet = ADMET(
                candidate_id=new_candidate.id,
                absorption=admet_data.get("absorption", {}).get("score", 0),
                distribution=admet_data.get("distribution", {}).get("score", 0),
                metabolism=admet_data.get("metabolism", {}).get("score", 0),
                excretion=admet_data.get("excretion", {}).get("score", 0),
                overall_score=admet_data.get("overall", 0),
                verdict=admet_data.get("verdict", "Unknown")
            )
            db.add(admet)
            
        added_count += 1

    db.commit()
    db.refresh(exp)
    return {
        "id": exp.id, 
        "message": "Experiment saved successfully",
        "new_candidates_added": added_count
    }


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
