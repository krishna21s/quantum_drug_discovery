"""
Image Rendering Routes
======================
FastAPI router for generating 2D structure images from SMILES using RDKit.
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from rdkit import Chem
from rdkit.Chem import Draw
import io

router = APIRouter(prefix="/api/image", tags=["image"])

@router.get("/render")
async def render_smiles(smiles: str = Query(..., description="SMILES string to render", min_length=1)):
    """
    Renders a 2D structure of the given SMILES string as a PNG image.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("RDKit could not parse SMILES")
            
        # Draw the molecule
        img = Draw.MolToImage(mol, size=(400, 300), fitImage=True, kekulize=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {str(e)}")
