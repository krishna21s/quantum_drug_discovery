"""
Similarity Analyzer API Routes
================================
FastAPI router for drug similarity analysis using Morgan fingerprints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .services.similarity_service import similarity_engine

router = APIRouter(prefix="/api/similarity", tags=["similarity"])


class AnalyzeRequest(BaseModel):
    smiles: str


@router.on_event("startup")
async def _load_db():
    similarity_engine.load()


@router.post("/analyze")
async def analyze_similarity(req: AnalyzeRequest):
    """
    Analyze a SMILES string against the reference drug database.
    Returns ranked therapeutic indications with matched drugs and similarity scores.
    """
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string is required")

    try:
        result = similarity_engine.analyze(smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return result.to_dict()
