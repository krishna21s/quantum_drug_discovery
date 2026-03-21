"""
Generator Service — On-demand Drug Candidate Generation
=========================================================
Loads the RL-trained ConditionedRNN and generates new
SMILES molecules conditioned on a protein pocket. Filters
for validity, drug-likeness, and scores with dual oracles.
"""

import os
import sys
import time
from typing import List, Optional

import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V4_DIR = os.path.join(_BACKEND_DIR, "construction_v4")
sys.path.insert(0, _V4_DIR)

import torch
from models.conditioned_rnn import ConditionedRNN
from training.pocket_conditioner import PocketConditioner
from config_v4 import V4_CHECKPOINT_DIR


class GeneratorService:
    """
    On-demand molecule generation using the RL-trained model.

    Pipeline:
        1. Load pocket φ for selected target (PDB ID)
        2. Sample N×2 raw SMILES (oversample for filtering)
        3. Filter for valid, drug-like molecules
        4. Score with binding oracle (XGB + QSVR)
        5. Return top-N ranked candidates
    """

    def __init__(self, binding_oracle=None):
        self._model = None
        self._pocket_conditioner = PocketConditioner(str(V4_CHECKPOINT_DIR))
        self._binding_oracle = binding_oracle
        self._device = "cpu"

        self._load_model()

    def _load_model(self):
        """Load the RL fine-tuned model."""
        model_path = os.path.join(str(V4_CHECKPOINT_DIR), "policy_egfr_rl.pt")
        if not os.path.exists(model_path):
            print(f"  [Generator] RL model not found: {model_path}")
            return

        self._model = ConditionedRNN.load(model_path, device=self._device)
        self._model.eval()
        print(f"  [Generator] RL model loaded ({sum(p.numel() for p in self._model.parameters()):,} params)")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def generate(
        self,
        pdb_id: str = "1M17",
        n_candidates: int = 20,
        temperature: float = 1.0,
        max_mw: float = 600.0,
    ) -> dict:
        """
        Generate new drug candidates for a given target.

        Args:
            pdb_id:        PDB identifier for the target protein
            n_candidates:  number of valid candidates to return
            temperature:   sampling temperature (higher = more diverse)
            max_mw:        maximum molecular weight filter

        Returns:
            dict with 'candidates' list and metadata
        """
        if self._model is None:
            raise RuntimeError("Generator model not loaded")

        t0 = time.time()
        n_candidates = min(max(n_candidates, 1), 100)
        temperature = max(0.5, min(temperature, 2.0))

        # 1. Get pocket phi for target
        phi = self._pocket_conditioner.load_or_compute(pdb_id)
        phi_tensor = torch.tensor(phi, dtype=torch.float32)

        # 2. Sample in batches until we have enough valid molecules
        #    Oversample significantly — model produces ~60-80% valid SMILES
        n_sample = max(n_candidates * 8, 100)

        all_raw = []
        for batch_start in range(0, n_sample, 64):
            batch_n = min(64, n_sample - batch_start)
            batch_smiles = self._model.sample_conditioned(
                phi_tensor, n=batch_n, temperature=temperature, device=self._device
            )
            all_raw.extend(batch_smiles)

        raw_smiles = all_raw

        # 3. Filter valid, drug-like molecules
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, Crippen

        valid_candidates = []
        seen = set()

        for smi in raw_smiles:
            if not smi or smi in seen:
                continue
            seen.add(smi)

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                continue
            seen.add(canonical)

            mw = Descriptors.MolWt(mol)
            if mw > max_mw or mw < 100:
                continue

            logp = Crippen.MolLogP(mol)
            qed_val = QED.qed(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

            candidate = {
                "smiles": canonical,
                "mw": round(mw, 2),
                "logp": round(logp, 2),
                "qed": round(qed_val, 3),
                "tpsa": round(tpsa, 1),
                "lipinski_pass": lipinski,
                "xgb_pic50": None,
                "quantum_pic50": None,
                "scoring_mode": None,
            }
            valid_candidates.append(candidate)

        # 4. Score with binding oracle
        if self._binding_oracle and valid_candidates:
            for cand in valid_candidates:
                try:
                    result = self._binding_oracle.score(cand["smiles"])
                    cand["xgb_pic50"] = result.get("xgb_pic50")
                    cand["quantum_pic50"] = result.get("pic50")
                    cand["scoring_mode"] = result.get("mode")
                except Exception:
                    pass

        # 5. Sort by best score and take top N
        def sort_key(c):
            qsvr = c.get("quantum_pic50") or 0
            xgb = c.get("xgb_pic50") or 0
            return max(qsvr, xgb)

        valid_candidates.sort(key=sort_key, reverse=True)
        top = valid_candidates[:n_candidates]

        # Assign ranks
        for i, c in enumerate(top, 1):
            c["rank"] = i

        # SA score (optional, computed after ranking)
        try:
            from rdkit.Chem import RDConfig
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            for c in top:
                mol = Chem.MolFromSmiles(c["smiles"])
                if mol:
                    c["sa_score"] = round(sascorer.calculateScore(mol), 2)
                else:
                    c["sa_score"] = 5.0
        except Exception:
            for c in top:
                c["sa_score"] = 3.0  # default

        elapsed = time.time() - t0

        return {
            "target": f"{pdb_id.upper()}",
            "n_requested": n_candidates,
            "n_sampled": n_sample,
            "n_valid": len(valid_candidates),
            "temperature": temperature,
            "generation_time_s": round(elapsed, 2),
            "candidates": top,
        }
