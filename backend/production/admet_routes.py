"""
ADMET API Routes
================
FastAPI endpoints for generating and retrieving ADMET property predictions.
Uses the ADMET-AI model via the admet_service module.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import SessionLocal
from database.models import Candidate, ADMET

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admet", tags=["ADMET"])


# ── Dependency ──────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic schemas ────────────────────────────────────────

class ADMETGenerateRequest(BaseModel):
    smiles: str


class ADMETResponse(BaseModel):
    absorption: float
    distribution: float
    metabolism: float
    excretion: float
    overall: float
    verdict: str
    candidate_id: int | None = None

    class Config:
        from_attributes = True


class ADMETBatchResponse(BaseModel):
    processed: int
    skipped: int
    failed: int
    message: str


# ── Routes ──────────────────────────────────────────────────

@router.post("/generate", response_model=ADMETResponse)
def generate_admet(req: ADMETGenerateRequest, db: Session = Depends(get_db)):
    """
    Generate ADMET predictions for a single SMILES string.
    If the SMILES corresponds to an existing candidate, store the result in DB.
    """
    from production.services.admet_service import generate_admet_from_smiles

    try:
        result = generate_admet_from_smiles(req.smiles)
    except Exception as e:
        logger.error(f"ADMET prediction failed for {req.smiles}: {e}")
        raise HTTPException(status_code=500, detail=f"ADMET prediction failed: {str(e)}")

    candidate_id = None

    # Check if candidate exists in DB
    candidate = db.query(Candidate).filter(Candidate.smiles == req.smiles).first()
    if candidate:
        # Check if ADMET entry already exists
        existing = db.query(ADMET).filter(ADMET.candidate_id == candidate.id).first()
        if existing:
            # Update existing entry
            existing.absorption = result["absorption"]
            existing.distribution = result["distribution"]
            existing.metabolism = result["metabolism"]
            existing.excretion = result["excretion"]
            existing.overall = result["overall"]
            existing.verdict = result["verdict"]
            db.commit()
            candidate_id = candidate.id
        else:
            # Create new ADMET entry
            admet_entry = ADMET(
                candidate_id=candidate.id,
                absorption=result["absorption"],
                distribution=result["distribution"],
                metabolism=result["metabolism"],
                excretion=result["excretion"],
                overall=result["overall"],
                verdict=result["verdict"],
            )
            db.add(admet_entry)
            db.commit()
            candidate_id = candidate.id

    return ADMETResponse(
        absorption=result["absorption"],
        distribution=result["distribution"],
        metabolism=result["metabolism"],
        excretion=result["excretion"],
        overall=result["overall"],
        verdict=result["verdict"],
        candidate_id=candidate_id,
    )


@router.post("/fill-missing", response_model=ADMETBatchResponse)
def fill_missing_admet(db: Session = Depends(get_db)):
    """
    Find all candidates WITHOUT an ADMET entry and generate predictions for them.
    This is a batch operation that may take a while for large datasets.
    """
    from production.services.admet_service import generate_admet_from_smiles

    # Find candidates that don't have an ADMET row
    candidates_without_admet = (
        db.query(Candidate)
        .outerjoin(ADMET, Candidate.id == ADMET.candidate_id)
        .filter(ADMET.id.is_(None))
        .all()
    )

    processed = 0
    skipped = 0
    failed = 0

    for candidate in candidates_without_admet:
        try:
            result = generate_admet_from_smiles(candidate.smiles)

            admet_entry = ADMET(
                candidate_id=candidate.id,
                absorption=result["absorption"],
                distribution=result["distribution"],
                metabolism=result["metabolism"],
                excretion=result["excretion"],
                overall=result["overall"],
                verdict=result["verdict"],
            )
            db.add(admet_entry)
            db.commit()
            processed += 1
            logger.info(f"ADMET generated for candidate {candidate.id} ({candidate.smiles[:30]}...)")
        except Exception as e:
            db.rollback()
            failed += 1
            logger.error(f"ADMET failed for candidate {candidate.id}: {e}")

    total = len(candidates_without_admet)
    return ADMETBatchResponse(
        processed=processed,
        skipped=skipped,
        failed=failed,
        message=f"Processed {processed}/{total} candidates. {failed} failed.",
    )


@router.get("/{candidate_id}", response_model=ADMETResponse)
def get_admet(candidate_id: int, db: Session = Depends(get_db)):
    """Retrieve ADMET data for a specific candidate."""
    admet = db.query(ADMET).filter(ADMET.candidate_id == candidate_id).first()
    if not admet:
        raise HTTPException(status_code=404, detail="ADMET data not found for this candidate.")

    return ADMETResponse(
        absorption=admet.absorption,
        distribution=admet.distribution,
        metabolism=admet.metabolism,
        excretion=admet.excretion,
        overall=admet.overall,
        verdict=admet.verdict,
        candidate_id=admet.candidate_id,
    )
