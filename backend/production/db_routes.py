from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import get_db
from database.models import Candidate, BindingAffinity, Toxicity

router = APIRouter(prefix="/api/db", tags=["Database"])

# Pydantic schemas for the response
class BindingAffinityBase(BaseModel):
    xgb_pic50: float | None = None
    qsvr_pic50: float | None = None
    scoring_mode: str | None = None
    class Config:
        from_attributes = True

class ToxicityBase(BaseModel):
    toxicity_score: float | None = None
    is_toxic: bool | None = None
    alerts_json: list | None = None
    class Config:
        from_attributes = True

class CandidateBase(BaseModel):
    id: int
    smiles: str
    target: str | None
    mw: float | None
    logp: float | None
    tpsa: float | None
    qed: float | None
    sa_score: float | None
    lipinski_pass: bool | None
    is_novel: bool | None
    binding_affinity: BindingAffinityBase | None = None
    toxicity: ToxicityBase | None = None

    class Config:
        from_attributes = True

@router.get("/candidates", response_model=list[CandidateBase])
def get_candidates(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Fetch all saved candidates."""
    candidates = db.query(Candidate).offset(skip).limit(limit).all()
    return candidates

@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Fetch a single candidate along with its related data."""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return {
        "candidate": {
            "id": candidate.id,
            "smiles": candidate.smiles,
            "target": candidate.target,
            "mw": candidate.mw,
            "logp": candidate.logp,
            "tpsa": candidate.tpsa,
            "qed": candidate.qed,
            "sa_score": candidate.sa_score,
            "lipinski_pass": candidate.lipinski_pass,
            "is_novel": candidate.is_novel
        },
        "binding_affinity": {
            "xgb_pic50": candidate.binding_affinity.xgb_pic50 if candidate.binding_affinity else None,
            "qsvr_pic50": candidate.binding_affinity.qsvr_pic50 if candidate.binding_affinity else None,
            "scoring_mode": candidate.binding_affinity.scoring_mode if candidate.binding_affinity else None,
        },
        "toxicity": {
            "toxicity_score": candidate.toxicity.toxicity_score if candidate.toxicity else None,
            "is_toxic": candidate.toxicity.is_toxic if candidate.toxicity else None,
            "alerts": candidate.toxicity.alerts_json if candidate.toxicity else None,
        }
    }


@router.post("/seed")
def seed_database(db: Session = Depends(get_db)):
    """
    Seed the database using the existing `final_candidates.json` checkpoint.
    This will populate Candidates, Binding Affinity, and simple Toxicity assumptions
    for the sake of completeness.
    """
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "construction_v4", "checkpoints", "final_candidates.json"
    )
    
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found")
        
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
        
    target = data.get("target", "Unknown")
    candidates_list = data.get("candidates", [])
    
    added_count = 0
    for c_data in candidates_list:
        smiles = c_data.get("smiles")
        if not smiles: continue
        
        # Check if exists
        existing = db.query(Candidate).filter(Candidate.smiles == smiles).first()
        if existing:
            continue
            
        # 1. Create Candidate
        new_candidate = Candidate(
            smiles=smiles,
            target=target,
            mw=c_data.get("mw"),
            logp=c_data.get("logp"),
            tpsa=c_data.get("tpsa"),
            qed=c_data.get("qed"),
            sa_score=c_data.get("sa_score"),
            lipinski_pass=c_data.get("lipinski_pass"),
            is_novel=c_data.get("is_novel")
        )
        db.add(new_candidate)
        db.flush() # flush to get an ID
        
        # 2. Add Binding Affinity
        xgb_score = c_data.get("xgb_pic50")
        quantum_score = c_data.get("quantum_pic50")
        if xgb_score is not None or quantum_score is not None:
            binding = BindingAffinity(
                candidate_id=new_candidate.id,
                xgb_pic50=xgb_score,
                qsvr_pic50=quantum_score,
                scoring_mode=c_data.get("scoring_mode", "qsvr_rbf")
            )
            db.add(binding)
            
        # 3. Add a placeholder Toxicty entry for testing (optional, some toxic, some safe based on QED)
        # Just rough approximations for the seed data so UI has something to show
        q = c_data.get("qed", 0.5)
        is_toxic = bool(q < 0.5) # simple heuristic for demo
        tox = Toxicity(
            candidate_id=new_candidate.id,
            canonical_smiles=smiles,
            toxicity_score=1.0 - q, # Inverse of QED loosely maps to toxicity risk for demo
            is_toxic=is_toxic,
            alerts_json=[{"name": "High Risk", "description": "Candidate triggered structural alerts"}] if is_toxic else []
        )
        db.add(tox)
        
        added_count += 1
        
    db.commit()
    
    return {"message": f"Successfully seeded {added_count} candidates into the database."}
