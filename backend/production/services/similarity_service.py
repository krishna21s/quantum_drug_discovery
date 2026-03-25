"""
Drug Similarity Service
========================
Uses RDKit Morgan fingerprints (ECFP4) and Tanimoto similarity to find
structurally similar approved drugs and predict therapeutic indications.

Approach:
  1. On startup, load a curated CSV of approved drugs with SMILES + indications.
  2. Pre-compute Morgan fingerprints (radius=2, 2048 bits) for all reference drugs.
  3. For a query SMILES, compute its fingerprint and calculate bulk Tanimoto
     similarity against all reference drugs.
  4. Group top matches by disease indication, returning ranked therapeutic areas.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ── Data classes ────────────────────────────────────────────────────

@dataclass
class MolecularProfile:
    molecular_weight: float
    logp: float
    tpsa: float
    hba: int
    hbd: int
    rotatable_bonds: int
    num_rings: int
    formula: str

    def to_dict(self):
        return asdict(self)


@dataclass
class DrugMatch:
    name: str
    similarity: float
    indication: str
    category: str
    target_protein: str
    mechanism: str

    def to_dict(self):
        return asdict(self)


@dataclass
class DiseaseResult:
    indication: str
    category: str
    max_similarity: float
    avg_similarity: float
    match_level: str          # CONFIRMED / HIGH / MODERATE
    matched_drugs: List[DrugMatch]
    top_target: str
    top_mechanism: str

    def to_dict(self):
        d = asdict(self)
        d["matched_drugs"] = [m.to_dict() for m in self.matched_drugs]
        return d


@dataclass
class SimilarityResult:
    query_smiles: str
    canonical_smiles: str
    molecular_profile: MolecularProfile
    diseases: List[DiseaseResult]
    total_diseases_found: int
    total_drugs_matched: int

    def to_dict(self):
        return {
            "query_smiles": self.query_smiles,
            "canonical_smiles": self.canonical_smiles,
            "molecular_profile": self.molecular_profile.to_dict(),
            "diseases": [d.to_dict() for d in self.diseases],
            "total_diseases_found": self.total_diseases_found,
            "total_drugs_matched": self.total_drugs_matched,
        }


# ── Reference Database ─────────────────────────────────────────────

@dataclass
class RefDrug:
    name: str
    smiles: str
    indication: str
    category: str
    target_protein: str
    mechanism: str
    fingerprint: DataStructs.ExplicitBitVect = field(repr=False, default=None)


class DrugSimilarityEngine:
    """Singleton-style engine. Call load() once at startup."""

    def __init__(self):
        self.ref_drugs: List[RefDrug] = []
        self._loaded = False

    def load(self, csv_path: str | None = None):
        if self._loaded:
            return

        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "drug_reference_db.csv"
            )

        if not os.path.exists(csv_path):
            logger.error(f"Reference DB not found: {csv_path}")
            return

        count = 0
        skipped = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    skipped += 1
                    logger.warning(f"Invalid SMILES for {row['name']}: {row['smiles']}")
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                drug = RefDrug(
                    name=row["name"],
                    smiles=row["smiles"],
                    indication=row["indication"],
                    category=row["category"],
                    target_protein=row["target_protein"],
                    mechanism=row["mechanism"],
                    fingerprint=fp,
                )
                self.ref_drugs.append(drug)
                count += 1

        self._loaded = True
        logger.info(f"Loaded {count} reference drugs ({skipped} skipped)")

    def analyze(self, smiles: str, min_similarity: float = 0.25) -> SimilarityResult:
        """
        Analyze a query SMILES against the reference database.

        Args:
            smiles: Query SMILES string
            min_similarity: Minimum Tanimoto threshold (0-1) for a match

        Returns:
            SimilarityResult with diseases ranked by max similarity
        """
        if not self._loaded:
            self.load()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: '{smiles}'")

        canonical = Chem.MolToSmiles(mol, canonical=True)

        # Compute query fingerprint
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

        # Compute molecular profile
        profile = MolecularProfile(
            molecular_weight=round(Descriptors.MolWt(mol), 1),
            logp=round(Descriptors.MolLogP(mol), 2),
            tpsa=round(Descriptors.TPSA(mol), 1),
            hba=rdMolDescriptors.CalcNumHBA(mol),
            hbd=rdMolDescriptors.CalcNumHBD(mol),
            rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
            num_rings=rdMolDescriptors.CalcNumRings(mol),
            formula=rdMolDescriptors.CalcMolFormula(mol),
        )

        # Bulk Tanimoto similarity
        ref_fps = [d.fingerprint for d in self.ref_drugs]
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)

        # Collect matches above threshold
        matches: list[DrugMatch] = []
        for i, sim in enumerate(similarities):
            if sim >= min_similarity:
                drug = self.ref_drugs[i]
                matches.append(DrugMatch(
                    name=drug.name,
                    similarity=round(sim, 4),
                    indication=drug.indication,
                    category=drug.category,
                    target_protein=drug.target_protein,
                    mechanism=drug.mechanism,
                ))

        # Group by indication
        indication_groups: dict[str, list[DrugMatch]] = {}
        for m in matches:
            if m.indication not in indication_groups:
                indication_groups[m.indication] = []
            indication_groups[m.indication].append(m)

        # Build disease results
        diseases: list[DiseaseResult] = []
        for indication, drug_matches in indication_groups.items():
            drug_matches.sort(key=lambda d: d.similarity, reverse=True)
            max_sim = drug_matches[0].similarity
            avg_sim = sum(d.similarity for d in drug_matches) / len(drug_matches)

            # Score as percentage (0-100)
            score = round(max_sim * 100)
            level = "CONFIRMED" if score >= 75 else "HIGH" if score >= 60 else "MODERATE"

            diseases.append(DiseaseResult(
                indication=indication,
                category=drug_matches[0].category,
                max_similarity=round(max_sim, 4),
                avg_similarity=round(avg_sim, 4),
                match_level=level,
                matched_drugs=drug_matches[:5],  # Top 5 per disease
                top_target=drug_matches[0].target_protein,
                top_mechanism=drug_matches[0].mechanism,
            ))

        # Sort by max similarity descending
        diseases.sort(key=lambda d: d.max_similarity, reverse=True)

        return SimilarityResult(
            query_smiles=smiles,
            canonical_smiles=canonical,
            molecular_profile=profile,
            diseases=diseases,
            total_diseases_found=len(diseases),
            total_drugs_matched=len(matches),
        )


# Module-level singleton
similarity_engine = DrugSimilarityEngine()
