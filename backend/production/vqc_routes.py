"""
API Routes — VQC Circuit Generation
=====================================
Generate real variational quantum circuits for specific molecules.
n_qubits and n_layers are derived from molecular complexity (not fixed at 8).
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/vqc", tags=["VQC Circuits"])


class GateInfo(BaseModel):
    type: str
    qubit: int
    col: int
    target: Optional[int] = None
    angle: Optional[float] = None
    label: Optional[str] = None


class CircuitResponse(BaseModel):
    smiles: str
    n_qubits: int
    n_layers: int
    circuit_depth: int
    total_gates: int
    total_parameters: int
    gates: list[GateInfo]
    feature_vector: list[float]
    gate_type_counts: dict[str, int]
    molecular_properties: dict


def _smiles_to_features(smiles: str):
    """Compute RDKit molecular descriptors and derive circuit parameters."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

        raw = np.array([mw, logp, tpsa, hbd, hba, rotatable, rings, heavy], dtype=np.float64)
        # Arctan normalization
        normalized = (2.0 / np.pi) * np.arctan(raw * 0.01)

        # n_qubits: maps heavy atoms to [4, 16], must be even
        n_qubits = min(16, max(4, int(heavy // 3)))
        if n_qubits % 2 != 0:
            n_qubits += 1

        # n_layers: driven by aromaticity (more rings = deeper circuit)
        n_layers = min(4, max(1, 1 + aromatic_rings))

        props = {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": int(hbd),
            "hba": int(hba),
            "rotatable_bonds": int(rotatable),
            "rings": int(rings),
            "aromatic_rings": int(aromatic_rings),
            "heavy_atoms": int(heavy),
            "stereocenters": int(stereo),
        }
        return normalized, n_qubits, n_layers, props

    except ImportError:
        import hashlib
        h = hashlib.sha256(smiles.encode()).digest()
        vals = np.array([b / 255.0 for b in h[:8]], dtype=np.float64)
        props = {k: 0 for k in ["mw","logp","tpsa","hbd","hba","rotatable_bonds","rings","aromatic_rings","heavy_atoms","stereocenters"]}
        return vals, 8, 2, props


def _build_circuit(features: np.ndarray, n_qubits: int, n_layers: int) -> list[dict]:
    """
    Data-reuploading VQC with CZ ring entanglement.
    Each qubit gets molecule-specific rotation angles.
    """
    gates = []
    col = 0

    for layer in range(n_layers):
        if layer == 0:
            for q in range(n_qubits):
                gates.append({"type": "H", "qubit": q, "col": col})
            col += 1

        # Data reuploading Ry — unique per qubit using two features
        for q in range(n_qubits):
            f0 = features[q % len(features)]
            f1 = features[(q + 1) % len(features)]
            angle = float((f0 + 0.5 * f1) * np.pi)
            gates.append({"type": "Ry", "qubit": q, "col": col,
                          "angle": round(angle, 4), "label": f"x[{q % len(features)}]"})
        col += 1

        # Parameterised Rz — shift per layer
        for q in range(n_qubits):
            f_idx = (q * 2 + layer + 1) % len(features)
            angle = float(features[f_idx] * np.pi * (0.5 + 0.1 * layer))
            gates.append({"type": "Rz", "qubit": q, "col": col,
                          "angle": round(angle, 4), "label": f"θ[{layer},{q}]"})
        col += 1

        # CZ even pairs
        for q in range(0, n_qubits - 1, 2):
            gates.append({"type": "CZ", "qubit": q, "col": col, "target": q + 1})
        col += 1

        # CZ odd pairs + wrap-around
        if n_qubits > 2:
            for q in range(1, n_qubits - 1, 2):
                gates.append({"type": "CZ", "qubit": q, "col": col, "target": q + 1})
            if n_qubits > 3:
                gates.append({"type": "CZ", "qubit": n_qubits - 1, "col": col, "target": 0})
            col += 1

    # Final rotation
    for q in range(n_qubits):
        f_idx = (q + n_layers) % len(features)
        angle = float(features[f_idx] * np.pi * 0.4)
        gates.append({"type": "Ry", "qubit": q, "col": col,
                      "angle": round(angle, 4), "label": f"θ_f[{q}]"})
    col += 1

    # Measurements
    for q in range(n_qubits):
        gates.append({"type": "M", "qubit": q, "col": col})

    return gates


@router.post("/circuit", response_model=CircuitResponse)
async def generate_circuit(body: dict):
    """Generate the real molecule-specific VQC circuit from a SMILES string."""
    smiles = body.get("smiles", "")
    if not smiles.strip():
        raise HTTPException(status_code=422, detail="SMILES string required")

    features, n_qubits, n_layers, props = _smiles_to_features(smiles.strip())

    # Allow manual override
    if body.get("n_qubits"):
        n_qubits = int(body["n_qubits"])
    if body.get("n_layers"):
        n_layers = int(body["n_layers"])

    gate_dicts = _build_circuit(features, n_qubits, n_layers)
    gates = [GateInfo(**g) for g in gate_dicts]

    type_counts: dict[str, int] = {}
    for g in gates:
        type_counts[g.type] = type_counts.get(g.type, 0) + 1

    depth = max(g.col for g in gates) + 1 if gates else 0
    n_params = sum(1 for g in gates if g.type in ("Ry", "Rz") and g.angle is not None)

    return CircuitResponse(
        smiles=smiles.strip(),
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_depth=depth,
        total_gates=len(gates),
        total_parameters=n_params,
        gates=gates,
        feature_vector=[round(float(f), 6) for f in features],
        gate_type_counts=type_counts,
        molecular_properties=props,
    )
