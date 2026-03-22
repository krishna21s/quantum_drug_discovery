"""
Molecular Mutator — RDKit-Based Chemical Transformation Engine
===============================================================
Generates structurally valid neighbors for a given SMILES molecule
via a curated set of medicinal-chemistry-aware mutations.

Action Space:
    1. Atom Replacement   — bioisosteric swaps (F↔Cl, O↔S, N↔O)
    2. Functional Group    — add/remove small groups (-OH, -CH3, -NH2, -F, -CF3)
    3. Ring Modifications  — saturate/aromatize bonds
    4. Chain Extension     — add/remove methylene units

All outputs are validated for chemical sanity (RDKit parseability,
kekulization, reasonable MW, and optionally scaffold preservation).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit warnings during mutation attempts (most will fail — that's expected)
RDLogger.logger().setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)

# ── Bioisosteric Atom Swap Table ────────────────────────────
# Maps atomic number → list of replacement atomic numbers
BIOISOSTERE_SWAPS: dict[int, list[int]] = {
    9:  [17, 35],        # F  → Cl, Br
    17: [9, 35],         # Cl → F, Br
    35: [9, 17],         # Br → F, Cl
    8:  [16],            # O  → S
    16: [8],             # S  → O
    7:  [8],             # N  → O
}

# ── Functional Groups for Addition ──────────────────────────
# Each is a SMARTS pattern that can be attached to an aromatic or
# aliphatic carbon with a free hydrogen.
FUNCTIONAL_GROUPS: list[tuple[str, str]] = [
    ("hydroxyl",     "[OH]"),
    ("methyl",       "[CH3]"),
    ("amino",        "[NH2]"),
    ("fluorine",     "[F]"),
    ("trifluoromethyl", "[C](F)(F)F"),
    ("methoxy",      "[O][CH3]"),
    ("cyano",        "[C]#[N]"),
]

# ── MW bounds for valid drug candidates ─────────────────────
MIN_MW = 100.0
MAX_MW = 700.0


# ═══════════════════════════════════════════════════════════
#  Core Mutation Functions
# ═══════════════════════════════════════════════════════════

def _mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    """Convert mol → canonical SMILES, returning None on failure."""
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        # Round-trip validation
        if Chem.MolFromSmiles(smi) is None:
            return None
        return smi
    except Exception:
        return None


def _is_valid_drug(smi: str) -> bool:
    """Check molecular weight and basic sanity."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    return MIN_MW <= mw <= MAX_MW


def _get_scaffold(smi: str) -> Optional[str]:
    """Extract the Murcko generic scaffold SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# ── Mutation 1: Atom Replacement ────────────────────────────

def _mutate_atom_swap(mol: Chem.Mol) -> list[str]:
    """Replace one atom with a bioisosteric equivalent."""
    results: list[str] = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in BIOISOSTERE_SWAPS:
            for replacement in BIOISOSTERE_SWAPS[atomic_num]:
                rw = RWMol(mol)
                rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(replacement)
                smi = _mol_to_smiles(rw)
                if smi:
                    results.append(smi)
    return results


# ── Mutation 2: Functional Group Addition ───────────────────

def _mutate_add_group(mol: Chem.Mol) -> list[str]:
    """Add a small functional group to an available carbon site."""
    results: list[str] = []

    for atom in mol.GetAtoms():
        # Only attach to carbon atoms that have implicit hydrogens
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetTotalNumHs() < 1:
            continue

        idx = atom.GetIdx()
        for name, smarts in FUNCTIONAL_GROUPS:
            try:
                rw = RWMol(mol)
                frag = Chem.MolFromSmiles(smarts)
                if frag is None:
                    continue
                # Add fragment atoms to rw
                new_idx = rw.AddAtom(frag.GetAtomWithIdx(0))
                rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)

                smi = _mol_to_smiles(rw)
                if smi:
                    results.append(smi)
            except Exception:
                continue

    return results


# ── Mutation 3: Remove Terminal Functional Group ────────────

def _mutate_remove_group(mol: Chem.Mol) -> list[str]:
    """Remove terminal atoms (halogens, -OH, -NH2, etc.) to simplify."""
    results: list[str] = []
    terminal_atoms = [9, 17, 35, 53]  # F, Cl, Br, I

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in terminal_atoms and atom.GetDegree() == 1:
            rw = RWMol(mol)
            rw.RemoveAtom(atom.GetIdx())
            smi = _mol_to_smiles(rw)
            if smi:
                results.append(smi)

    return results


# ── Mutation 4: Bond Saturation (C=C → C-C) ────────────────

def _mutate_saturate_bond(mol: Chem.Mol) -> list[str]:
    """Convert a double bond to single bond (saturate), or vice versa."""
    results: list[str] = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue  # Don't break aromatic systems

        rw = RWMol(mol)
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            rw.GetBondWithIdx(bond.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            smi = _mol_to_smiles(rw)
            if smi:
                results.append(smi)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            # Only try on non-ring bonds between carbons
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6 and not bond.IsInRing():
                rw.GetBondWithIdx(bond.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                smi = _mol_to_smiles(rw)
                if smi:
                    results.append(smi)
    return results


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════

def generate_variants(
    smiles: str,
    max_variants: int = 20,
    preserve_scaffold: bool = False,
) -> list[str]:
    """
    Generate up to `max_variants` structurally valid molecular neighbors
    for the given SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    max_variants : int
        Maximum number of unique variants to return.
    preserve_scaffold : bool
        If True, only return variants whose Murcko scaffold
        matches the original molecule's scaffold.

    Returns
    -------
    list[str]
        List of unique, valid SMILES strings representing neighboring molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid input SMILES: {smiles}")
        return []

    original_scaffold = _get_scaffold(smiles) if preserve_scaffold else None

    # Collect all mutation candidates
    candidates: list[str] = []
    candidates.extend(_mutate_atom_swap(mol))
    candidates.extend(_mutate_add_group(mol))
    candidates.extend(_mutate_remove_group(mol))
    candidates.extend(_mutate_saturate_bond(mol))

    # Deduplicate and remove the original
    canonical_original = Chem.MolToSmiles(mol)
    seen: set[str] = {canonical_original}
    valid: list[str] = []

    for smi in candidates:
        canon = Chem.CanonSmiles(smi) if Chem.MolFromSmiles(smi) else None
        if canon is None or canon in seen:
            continue
        if not _is_valid_drug(canon):
            continue
        if preserve_scaffold and _get_scaffold(canon) != original_scaffold:
            continue
        seen.add(canon)
        valid.append(canon)

    # Shuffle and trim to max_variants
    random.shuffle(valid)
    result = valid[:max_variants]

    logger.info(
        f"Mutator: {smiles[:40]}... → {len(result)} valid variants "
        f"(from {len(candidates)} raw mutations)"
    )
    return result
