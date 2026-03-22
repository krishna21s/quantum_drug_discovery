"""
Refinement Environment — Delta-Reward Lead Optimization
=========================================================
Given a starting molecule, iteratively mutates it using the RDKit Mutator
and selects the variant with the highest delta-reward at each step.

Reward Signal (multi-objective, quantum-enhanced):
    ΔR = w1·Δ(Ensemble_Binding) + w2·Δ(ADMET_overall) − w3·Δ(Ensemble_Tox) − w4·Δ(SA)

    Ensemble_Binding = 0.55*XGB + 0.45*QSVR  ← quantum electronic features
    Ensemble_Tox     = avg(XGB_prob, QSVM_prob)  ← hybrid quantum safety

The optimizer runs a greedy best-first search:
    Step 1: Generate N variants of current molecule
    Step 2: Score all variants (ADMET + Binding + Toxicity)
    Step 3: Pick the one with the best delta-reward
    Step 4: If improved → accept as new current molecule, else → stop
    Repeat for max_steps or until convergence.

Returns the full trajectory so the frontend can display the evolution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from .mutator import generate_variants

logger = logging.getLogger(__name__)


# ── Reward Weights ──────────────────────────────────────────
# Tunable hyperparameters for the multi-objective reward
W_BINDING   = 0.35   # Binding affinity (pIC50)
W_ADMET     = 0.30   # ADMET overall score
W_TOXICITY  = 0.25   # Toxicity (lower = better)
W_DRUGLIKE  = 0.10   # QED drug-likeness


@dataclass
class MoleculeScore:
    """Scorecard for a single molecule at one point in time."""
    smiles: str
    # Binding affinity — dual oracle
    xgb_pic50: float = 0.0
    qsvr_pic50: float = 0.0
    ensemble_pic50: float = 0.0       # 0.55*XGB + 0.45*QSVR
    # ADMET
    admet_overall: float = 0.0
    admet_absorption: float = 0.0
    admet_distribution: float = 0.0
    admet_metabolism: float = 0.0
    admet_excretion: float = 0.0
    admet_verdict: str = ""
    # Toxicity — dual oracle
    toxicity_xgb: float = 0.5         # classical XGB probability
    toxicity_quantum: float = 0.5     # quantum SVM probability
    toxicity_ensemble: float = 0.5    # avg(XGB, quantum)
    toxicity_prob: float = 0.5        # backward compat
    # Drug-likeness
    qed_score: float = 0.0
    sa_score: float = 0.0
    mw: float = 0.0
    composite_reward: float = 0.0


@dataclass
class RefinementStep:
    """One step in the optimization trajectory."""
    step: int
    smiles: str
    scores: MoleculeScore
    delta_reward: float = 0.0
    variants_evaluated: int = 0
    accepted: bool = True
    mutation_type: str = ""


@dataclass
class RefinementResult:
    """Complete result of an optimization run."""
    original_smiles: str
    final_smiles: str
    trajectory: list[dict] = field(default_factory=list)
    total_steps: int = 0
    total_improvement: float = 0.0
    elapsed_seconds: float = 0.0
    converged: bool = False


# ═══════════════════════════════════════════════════════════
#  Scoring Functions
# ═══════════════════════════════════════════════════════════

def _score_molecule(
    smiles: str,
    binding_oracle: Any = None,
    toxicity_pipeline: Any = None,
) -> MoleculeScore:
    """
    Score a molecule across all objectives.
    Uses whatever oracles are available; gracefully degrades if some are missing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MoleculeScore(smiles=smiles)

    score = MoleculeScore(smiles=smiles)

    # ── Molecular properties (always available via RDKit) ────
    score.mw = Descriptors.MolWt(mol)
    score.qed_score = QED.qed(mol)

    # SA score (Synthetic Accessibility) — requires sascorer
    try:
        from rdkit.Contrib.SA_Score import sascorer
        score.sa_score = sascorer.calculateScore(mol) / 10.0  # normalize to 0-1
    except Exception:
        score.sa_score = 0.5  # neutral default

    # ── ADMET scoring (via admet_service) ────────────────────
    try:
        from production.services.admet_service import generate_admet_from_smiles
        admet = generate_admet_from_smiles(smiles)
        score.admet_overall = admet["overall"]
        score.admet_absorption = admet["absorption"]
        score.admet_distribution = admet["distribution"]
        score.admet_metabolism = admet["metabolism"]
        score.admet_excretion = admet["excretion"]
        score.admet_verdict = admet["verdict"]
    except Exception as e:
        logger.warning(f"ADMET scoring failed for {smiles[:30]}: {e}")
        score.admet_overall = 0.5

    # ── Binding affinity — XGBoost + Quantum SVR ensemble ────
    # Both oracles provide pIC50 predictions. Ensembling them gives
    # a signal that captures fingerprint patterns (XGB) AND quantum
    # electronic structure features (QSVR).
    if binding_oracle is not None:
        try:
            result = binding_oracle.score(smiles)
            score.xgb_pic50 = result.get("xgb_pic50", 0.0) or 0.0
            score.qsvr_pic50 = result.get("pic50", 0.0) or 0.0

            # Ensemble: weighted average (XGB more stable, QSVR adds quantum depth)
            if score.xgb_pic50 > 0 and score.qsvr_pic50 > 0:
                score.ensemble_pic50 = 0.55 * score.xgb_pic50 + 0.45 * score.qsvr_pic50
            elif score.xgb_pic50 > 0:
                score.ensemble_pic50 = score.xgb_pic50
            elif score.qsvr_pic50 > 0:
                score.ensemble_pic50 = score.qsvr_pic50

            logger.debug(
                f"Binding ensemble: XGB={score.xgb_pic50:.2f}, "
                f"QSVR={score.qsvr_pic50:.2f} -> ensemble={score.ensemble_pic50:.2f}"
            )
        except Exception as e:
            logger.warning(f"Binding scoring failed for {smiles[:30]}: {e}")

    # ── Toxicity — XGBoost + Quantum SVM ensemble ────────────
    # predict_fast returns classical (XGB) and quantum (kernel SVM)
    # probabilities. We capture both for a richer safety signal.
    if toxicity_pipeline is not None:
        try:
            result = toxicity_pipeline.predict_fast(smiles)
            score.toxicity_xgb = result.get("xgb_prob", 0.5)
            score.toxicity_quantum = result.get("quantum_prob", 0.5)
            score.toxicity_ensemble = result.get("ensemble_prob", 0.5)
            score.toxicity_prob = score.toxicity_ensemble  # backward compat

            logger.debug(
                f"Toxicity ensemble: XGB={score.toxicity_xgb:.3f}, "
                f"QSVM={score.toxicity_quantum:.3f} -> ensemble={score.toxicity_ensemble:.3f}"
            )
        except Exception as e:
            logger.warning(f"Toxicity scoring failed for {smiles[:30]}: {e}")

    # ── Composite reward (quantum-enhanced) ──────────────────
    # Uses ENSEMBLE binding (XGB+QSVR) and ENSEMBLE toxicity (XGB+QSVM)
    binding_norm = min(max((score.ensemble_pic50 - 4.0) / 6.0, 0.0), 1.0)
    safety_norm = 1.0 - score.toxicity_ensemble

    score.composite_reward = (
        W_BINDING  * binding_norm
        + W_ADMET  * score.admet_overall
        + W_TOXICITY * safety_norm
        + W_DRUGLIKE * score.qed_score
    )

    return score


def _compute_delta(new: MoleculeScore, old: MoleculeScore) -> float:
    """Compute the delta-reward between two molecule scores."""
    return new.composite_reward - old.composite_reward


# ═══════════════════════════════════════════════════════════
#  Core Optimization Loop
# ═══════════════════════════════════════════════════════════

def optimize_candidate(
    smiles: str,
    max_steps: int = 5,
    variants_per_step: int = 15,
    preserve_scaffold: bool = True,
    min_improvement: float = 0.005,
    binding_oracle: Any = None,
    toxicity_pipeline: Any = None,
) -> RefinementResult:
    """
    Iteratively refine a molecule to improve its multi-objective score.

    Parameters
    ----------
    smiles : str
        Starting SMILES string.
    max_steps : int
        Maximum number of refinement iterations.
    variants_per_step : int
        Number of structural variants to generate and evaluate per step.
    preserve_scaffold : bool
        If True, only mutate side chains (Murcko scaffold is frozen).
    min_improvement : float
        Minimum delta-reward to accept a step. Below this, we converge.
    binding_oracle : Any
        The QuantumOracle instance (from app.state.binding_oracle).
    toxicity_pipeline : Any
        The Toxicity pipeline (from app.state.pipeline).

    Returns
    -------
    RefinementResult
        Full trajectory of the optimization.
    """
    t0 = time.time()

    # Score the starting molecule
    current_smiles = smiles
    current_score = _score_molecule(current_smiles, binding_oracle, toxicity_pipeline)

    trajectory: list[RefinementStep] = [
        RefinementStep(
            step=0,
            smiles=current_smiles,
            scores=current_score,
            delta_reward=0.0,
            variants_evaluated=0,
            accepted=True,
            mutation_type="original",
        )
    ]

    converged = False

    for step in range(1, max_steps + 1):
        logger.info(f"[Refine] Step {step}/{max_steps} — current reward: {current_score.composite_reward:.4f}")

        # Generate variants
        variants = generate_variants(
            current_smiles,
            max_variants=variants_per_step,
            preserve_scaffold=preserve_scaffold,
        )

        if not variants:
            logger.info(f"[Refine] No valid variants generated at step {step}. Stopping.")
            converged = True
            break

        # Score all variants and find the best
        best_variant: Optional[str] = None
        best_score: Optional[MoleculeScore] = None
        best_delta: float = -float("inf")

        for v_smiles in variants:
            v_score = _score_molecule(v_smiles, binding_oracle, toxicity_pipeline)
            delta = _compute_delta(v_score, current_score)
            if delta > best_delta:
                best_delta = delta
                best_variant = v_smiles
                best_score = v_score

        # Accept or reject
        if best_delta >= min_improvement and best_variant and best_score:
            current_smiles = best_variant
            current_score = best_score

            trajectory.append(RefinementStep(
                step=step,
                smiles=current_smiles,
                scores=current_score,
                delta_reward=round(best_delta, 6),
                variants_evaluated=len(variants),
                accepted=True,
                mutation_type="improved",
            ))
            logger.info(f"[Refine] Step {step} ACCEPTED — Δ={best_delta:.4f}, new reward={current_score.composite_reward:.4f}")
        else:
            # No improvement found → converge
            trajectory.append(RefinementStep(
                step=step,
                smiles=current_smiles,
                scores=current_score,
                delta_reward=round(best_delta, 6) if best_delta > -float("inf") else 0.0,
                variants_evaluated=len(variants),
                accepted=False,
                mutation_type="converged",
            ))
            converged = True
            logger.info(f"[Refine] Step {step} CONVERGED — best Δ={best_delta:.4f} < threshold {min_improvement}")
            break

    elapsed = time.time() - t0
    total_improvement = current_score.composite_reward - trajectory[0].scores.composite_reward

    return RefinementResult(
        original_smiles=smiles,
        final_smiles=current_smiles,
        trajectory=[asdict(step) for step in trajectory],
        total_steps=len(trajectory) - 1,
        total_improvement=round(total_improvement, 6),
        elapsed_seconds=round(elapsed, 2),
        converged=converged,
    )
