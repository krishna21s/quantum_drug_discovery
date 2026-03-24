"""
Pydantic Schemas — Candidates & Binding Affinity
==================================================
Request/response models for the candidate generation and
binding affinity scoring endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Candidate Models ────────────────────────────────────────

class ADMETScores(BaseModel):
    """ADMET prediction scores for a candidate."""
    absorption: float = 0.0
    distribution: float = 0.0
    metabolism: float = 0.0
    excretion: float = 0.0
    overall: float = 0.0
    verdict: str = "Unknown"


class CandidateItem(BaseModel):
    """Single drug candidate from RL fine-tuning."""
    rank: int
    smiles: str
    xgb_pic50: float = Field(description="XGBoost predicted pIC50")
    quantum_pic50: Optional[float] = Field(description="QSVR predicted pIC50")
    qed: float = Field(description="Quantitative Estimate of Drug-likeness (0-1)")
    sa_score: float = Field(description="Synthetic Accessibility score (1=easy, 10=hard)")
    mw: float = Field(description="Molecular weight (Da)")
    logp: float = Field(description="Lipophilicity LogP")
    lipinski_pass: bool = Field(description="Passes Lipinski's Rule of Five")
    tpsa: Optional[float] = Field(default=None, description="Topological Polar Surface Area")
    is_novel: Optional[bool] = Field(default=None, description="Not found in training set")
    scoring_mode: Optional[str] = Field(default=None, description="QSVR scoring mode used")
    docking_score: Optional[float] = Field(default=None, description="Estimated docking score (kcal/mol)")
    admet: Optional[ADMETScores] = Field(default=None, description="ADMET prediction scores")


class CandidatesListResponse(BaseModel):
    """Response for GET /api/candidates."""
    target: str = Field(description="Target protein (e.g. 'EGFR (PDB 1M17)')")
    n_rl_episodes: int = Field(description="Number of RL episodes used")
    total_generated: int = Field(description="Total molecules generated during RL")
    final_reward: float = Field(description="Final RL reward value")
    total_time_min: float = Field(description="Total RL training time (minutes)")
    candidates: list[CandidateItem]


# ── Binding Affinity Models ─────────────────────────────────

class BindingScoreRequest(BaseModel):
    """Score a single SMILES for binding affinity."""
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="SMILES string of the molecule to score",
        json_schema_extra={"examples": ["c1ccc(Nc2ncnc3ccccc23)cn1"]},
    )


class BindingScoreResponse(BaseModel):
    """Binding affinity prediction result."""
    smiles: str
    canonical_smiles: Optional[str] = None
    xgb_pic50: Optional[float] = Field(description="XGBoost predicted pIC50")
    qsvr_pic50: Optional[float] = Field(description="QSVR predicted pIC50")
    scoring_mode: str = Field(description="Scoring mode: 'qsvr_rbf', 'qsvr_hybrid', 'xgb_fallback'")
    latency_s: float = Field(description="Scoring latency in seconds")
    error: Optional[str] = None


class BatchBindingRequest(BaseModel):
    """Batch binding affinity scoring request."""
    smiles_list: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of SMILES strings (max 50)",
    )


class BatchBindingResponse(BaseModel):
    """Batch binding affinity results."""
    predictions: list[BindingScoreResponse]
    summary: dict = Field(description="Aggregate statistics")
    total_time_s: float


# ── Generation Models ───────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request to generate new drug candidates."""
    pdb_id: str = Field(
        default="1M17",
        description="PDB identifier for the target protein (e.g. 1M17 for EGFR)",
    )
    n_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidates to generate (1-100)",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Sampling temperature (0.5=conservative, 2.0=creative)",
    )
    max_mw: float = Field(
        default=600.0,
        description="Maximum molecular weight filter (Da)",
    )
    stress_factors: list[str] = Field(
        default=[],
        description="Stress modifiers to apply: 'mutation', 'folding', 'thermal', 'binding'",
    )
    docking_engine: str = Field(
        default="autodock_vina",
        description="Docking engine: 'autodock_vina', 'gnina', 'none'",
    )
    run_admet: bool = Field(
        default=True,
        description="Run ADMET predictions on generated candidates",
    )
    vqe_optimizer: str = Field(
        default="COBYLA",
        description="VQE optimizer: COBYLA, SPSA, L-BFGS-B",
    )
    vqe_max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Max VQE iterations",
    )


class GenerateResponse(BaseModel):
    """Response from candidate generation."""
    target: str
    n_requested: int
    n_sampled: int
    n_valid: int
    temperature: float
    generation_time_s: float
    stress_applied: list[str] = Field(default=[], description="Stress factors that were applied")
    docking_engine: str = Field(default="none", description="Docking engine used")
    vqe_optimizer: str = Field(default="COBYLA", description="VQE optimizer used")
    vqe_max_iterations: int = Field(default=100, description="Max VQE iterations")
    candidates: list[CandidateItem]
